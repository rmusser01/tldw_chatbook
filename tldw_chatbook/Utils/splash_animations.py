# splash_animations.py
# Animation effects library for splash screens
# Provides various animation effects like Matrix rain, glitch, typewriter, etc.

import random
import time
import math
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass
from rich.text import Text
from rich.style import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align

from loguru import logger

from tldw_chatbook.Utils.Splash_Strings import splashscreen_message_selection

# Constants for escaping Rich markup
ESCAPED_OPEN_BRACKET = '\['
ESCAPED_CLOSE_BRACKET = '\]'


@dataclass
class AnimationFrame:
    """Represents a single frame of animation."""
    content: str
    style: Optional[Style] = None
    duration: float = 0.1


class BaseEffect:
    """Base class for animation effects."""
    
    def __init__(self, parent_widget: Any, **kwargs):
        self.parent = parent_widget
        self.frame_count = 0
        self.start_time = time.time()
        
    def update(self) -> Optional[str]:
        """Update and return the next frame of animation."""
        self.frame_count += 1
        return None
    
    def reset(self) -> None:
        """Reset the animation to its initial state."""
        self.frame_count = 0
        self.start_time = time.time()


class MatrixRainEffect(BaseEffect):
    """Matrix-style falling characters effect."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = (f"Loading user interface...{splashscreen_message_selection}"),
        width: int = 80,
        height: int = 24,
        speed: float = 0.05,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        self.speed = speed
        
        # Matrix rain state
        self.columns: List[List[Tuple[str, int]]] = []
        self.column_speeds: List[float] = []
        self.matrix_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Initialize columns
        self._init_columns()
        
        # Title reveal progress
        self.title_reveal_progress = 0.0
        self.subtitle_reveal_progress = 0.0
    
    def _init_columns(self) -> None:
        """Initialize the matrix rain columns."""
        self.columns = []
        self.column_speeds = []
        
        for _ in range(self.width):
            # Each column is a list of (character, brightness) tuples
            column = []
            column_height = random.randint(5, 15)
            
            for i in range(column_height):
                char = random.choice(self.matrix_chars)
                brightness = max(0, 255 - (i * 20))
                column.append((char, brightness))
            
            self.columns.append(column)
            self.column_speeds.append(random.uniform(0.3, 1.0))
    
    def update(self) -> Optional[str]:
        """Update the matrix rain effect."""
        # Create the display buffer
        display = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        styles = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Update and render columns
        for col_idx, column in enumerate(self.columns):
            # Move column down based on speed
            if self.frame_count % int(1 / self.column_speeds[col_idx]) == 0:
                # Add new character at top
                new_char = random.choice(self.matrix_chars)
                column.insert(0, (new_char, 255))
                
                # Remove old characters at bottom
                if len(column) > self.height:
                    column.pop()
                
                # Fade existing characters
                for i in range(len(column)):
                    char, brightness = column[i]
                    new_brightness = max(0, brightness - 25)
                    column[i] = (char, new_brightness)
            
            # Render column to display
            for row_idx, (char, brightness) in enumerate(column):
                if row_idx < self.height and brightness > 0:
                    display[row_idx][col_idx] = char
                    # Create style based on brightness
                    green_value = int(brightness)
                    styles[row_idx][col_idx] = f"rgb(0,{green_value},0)"
        
        # Overlay title and subtitle with reveal effect
        elapsed = time.time() - self.start_time
        
        # Start revealing title after 0.5 seconds
        if elapsed > 0.5:
            self.title_reveal_progress = min(1.0, (elapsed - 0.5) / 1.0)
            title_len = int(len(self.title) * self.title_reveal_progress)
            title_to_show = self.title[:title_len]
            
            # Center the title
            title_row = self.height // 2 - 2
            title_col = (self.width - len(self.title)) // 2
            
            for i, char in enumerate(title_to_show):
                if title_col + i < self.width:
                    display[title_row][title_col + i] = char
                    styles[title_row][title_col + i] = "bold rgb(0,255,0)"
        
        # Start revealing subtitle after 1.0 seconds
        if elapsed > 1.0:
            self.subtitle_reveal_progress = min(1.0, (elapsed - 1.0) / 1.0)
            subtitle_len = int(len(self.subtitle) * self.subtitle_reveal_progress)
            subtitle_to_show = self.subtitle[:subtitle_len]
            
            # Center the subtitle
            subtitle_row = self.height // 2
            subtitle_col = (self.width - len(self.subtitle)) // 2
            
            for i, char in enumerate(subtitle_to_show):
                if subtitle_col + i < self.width:
                    display[subtitle_row][subtitle_col + i] = char
                    styles[subtitle_row][subtitle_col + i] = "rgb(0,200,0)"
        
        # Convert to Rich markup format with proper escaping
        output_lines = []
        for row_idx, row in enumerate(display):
            line_chars = []
            current_style = None
            current_text = ""
            
            for col_idx, char in enumerate(row):
                style = styles[row_idx][col_idx] or "rgb(0,50,0)"
                
                # Escape Rich markup special characters
                escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET).replace(']', ESCAPED_CLOSE_BRACKET)
                
                if style != current_style:
                    # Close previous style and add text
                    if current_style and current_text:
                        line_chars.append(f"[{current_style}]{current_text}[/{current_style}]")
                    current_style = style
                    current_text = escaped_char
                else:
                    current_text += escaped_char
            
            # Add remaining text
            if current_style and current_text:
                line_chars.append(f"[{current_style}]{current_text}[/{current_style}]")
            
            output_lines.append(''.join(line_chars))
        
        return '\n'.join(output_lines)


class GlitchEffect(BaseEffect):
    """Glitch/corruption effect for text."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        glitch_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?",
        intensity: float = 0.3,
        speed: float = 0.1,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.original_content = content
        self.glitch_chars = glitch_chars
        self.intensity = intensity
        self.speed = speed
        
        # Parse content into lines
        self.lines = content.strip().split('\n')
        self.glitch_positions: List[Tuple[int, int]] = []
        self.color_shift = 0
    
    def update(self) -> Optional[str]:
        """Update the glitch effect."""
        # Randomly select positions to glitch
        if self.frame_count % 3 == 0:  # Update glitch positions every 3 frames
            self.glitch_positions = []
            for _ in range(int(len(self.lines) * self.intensity)):
                row = random.randint(0, len(self.lines) - 1)
                col = random.randint(0, max(0, len(self.lines[row]) - 1))
                self.glitch_positions.append((row, col))
        
        # Apply glitch effect
        glitched_lines = []
        for row_idx, line in enumerate(self.lines):
            glitched_line = list(line)
            
            # Apply glitches to this line
            for glitch_row, glitch_col in self.glitch_positions:
                if glitch_row == row_idx and glitch_col < len(glitched_line):
                    # Replace with glitch character
                    glitched_line[glitch_col] = random.choice(self.glitch_chars)
            
            glitched_lines.append(''.join(glitched_line))
        
        # Create color-shifted text with Rich markup
        output_lines = []
        self.color_shift = (self.color_shift + 10) % 360
        
        for line in glitched_lines:
            # Escape Rich markup special characters
            escaped_line = line.replace('[', r'\[').replace(']', r'\]')
            
            # Random color shifts for glitch effect
            if random.random() < 0.1:  # 10% chance of color shift
                r = random.randint(100, 255)
                g = random.randint(0, 100)
                b = random.randint(0, 100)
                # Use Rich markup format
                output_lines.append(f"[bold rgb({r},{g},{b})]{escaped_line}[/bold rgb({r},{g},{b})]")
            else:
                output_lines.append(f"[bold white]{escaped_line}[/bold white]")
        
        return '\n'.join(output_lines)


class TypewriterEffect(BaseEffect):
    """Typewriter effect that reveals text character by character."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        speed: float = 0.05,
        cursor: str = "█",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.speed = speed
        self.cursor = cursor
        self.chars_revealed = 0
        self.total_chars = len(content)
        self.cursor_blink = True
    
    def update(self) -> Optional[str]:
        """Update the typewriter effect."""
        # Calculate how many characters to reveal
        elapsed = time.time() - self.start_time
        chars_to_reveal = int(elapsed / self.speed)
        self.chars_revealed = min(chars_to_reveal, self.total_chars)
        
        # Get the revealed portion
        revealed_text = self.content[:self.chars_revealed]
        
        # Add cursor if not done
        if self.chars_revealed < self.total_chars:
            # Blink cursor
            if self.frame_count % 10 < 5:
                revealed_text += self.cursor
        
        return revealed_text


class FadeEffect(BaseEffect):
    """Fade in/out effect for content."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        fade_in: bool = True,
        duration: float = 1.0,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.fade_in = fade_in
        self.duration = duration
    
    def update(self) -> Optional[str]:
        """Update the fade effect."""
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / self.duration)
        
        if not self.fade_in:
            progress = 1.0 - progress
        
        # Calculate opacity (0-255)
        opacity = int(progress * 255)
        
        # Create faded text with Rich markup
        lines = self.content.split('\n')
        output_lines = []
        
        for line in lines:
            # Escape Rich markup special characters
            escaped_line = line.replace('[', r'\[').replace(']', r'\]')
            # Use Rich markup format
            output_lines.append(f"[rgb({opacity},{opacity},{opacity})]{escaped_line}[/rgb({opacity},{opacity},{opacity})]")
        
        return '\n'.join(output_lines)


class RetroTerminalEffect(BaseEffect):
    """Retro CRT terminal effect with scanlines and phosphor glow."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        scanline_speed: float = 0.02,
        phosphor_glow: bool = True,
        flicker: bool = True,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.scanline_speed = scanline_speed
        self.phosphor_glow = phosphor_glow
        self.flicker = flicker
        
        self.lines = content.strip().split('\n')
        self.scanline_position = 0
        self.flicker_intensity = 1.0
    
    def update(self) -> Optional[str]:
        """Update the retro terminal effect."""
        # Update scanline position
        self.scanline_position = (self.scanline_position + 1) % (len(self.lines) + 10)
        
        # Random flicker
        if self.flicker and random.random() < 0.05:
            self.flicker_intensity = random.uniform(0.8, 1.0)
        else:
            self.flicker_intensity = min(1.0, self.flicker_intensity + 0.02)
        
        # Build display with Rich markup
        output_lines = []
        
        for idx, line in enumerate(self.lines):
            # Calculate line brightness based on scanline
            distance_from_scanline = abs(idx - self.scanline_position)
            scanline_brightness = max(0.3, 1.0 - (distance_from_scanline * 0.1))
            
            # Apply flicker
            brightness = scanline_brightness * self.flicker_intensity
            
            # Create phosphor glow effect
            if self.phosphor_glow:
                # Green phosphor color with brightness
                r = int(0 * brightness)
                g = int(255 * brightness)
                b = int(100 * brightness)  # Slight blue tint
                color = f"rgb({r},{g},{b})"
            else:
                # Monochrome
                gray = int(255 * brightness)
                color = f"rgb({gray},{gray},{gray})"
            
            # Escape Rich markup special characters
            escaped_line = line.replace('[', r'\[').replace(']', r'\]')
            
            # Add scanline effect
            if distance_from_scanline < 2:
                # Brighter scanline with bold
                output_lines.append(f"[bold {color}]{escaped_line}[/bold {color}]")
            else:
                output_lines.append(f"[{color}]{escaped_line}[/{color}]")
        
        # Add some noise/static at the bottom
        if random.random() < 0.1:
            static_line = ''.join(random.choice(' .:-=+*#%@') for _ in range(len(self.lines[0]) if self.lines else 40))
            # Escape any brackets in static line
            escaped_static = static_line.replace('[', r'\[').replace(']', r'\]')
            output_lines.append(f"[rgb(50,50,50)]{escaped_static}[/rgb(50,50,50)]")
        
        return '\n'.join(output_lines)


class DigitalRainEffect(BaseEffect):
    """Digital rain effect with varied characters and color options."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Interface Loading...",
        width: int = 80,
        height: int = 24,
        speed: float = 0.05, # Interval for updates
        base_chars: str = "abcdefghijklmnopqrstuvwxyz0123456789",
        highlight_chars: str = "!@#$%^&*()_+=-{}[]|:;<>,.?/~",
        base_color: str = "dim green", # Rich style for base rain
        highlight_color: str = "bold green", # Rich style for highlighted chars
        title_style: str = "bold white",
        subtitle_style: str = "white",
        highlight_chance: float = 0.1, # Chance for a character to be a highlight_char
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        self.speed = speed # Not directly used in update logic timing, but for animation timer

        self.all_chars = base_chars + highlight_chars
        self.base_chars = base_chars
        self.highlight_chars = highlight_chars
        self.base_color = base_color
        self.highlight_color = highlight_color
        self.title_style = title_style
        self.subtitle_style = subtitle_style
        self.highlight_chance = highlight_chance

        self.columns: List[List[Tuple[str, str]]] = [] # char, style
        self.column_speeds: List[float] = [] # How many frames until this column updates
        self.column_next_update: List[int] = [] # Frame counter for next update

        self._init_columns()

        self.title_reveal_progress = 0.0
        self.subtitle_reveal_progress = 0.0

    def _init_columns(self) -> None:
        self.columns = []
        self.column_speeds = []
        self.column_next_update = []

        for _ in range(self.width):
            column = []
            # Initial column population (sparse)
            for _ in range(random.randint(self.height // 4, self.height // 2)):
                char = random.choice(self.all_chars)
                style = self.highlight_color if char in self.highlight_chars or random.random() < self.highlight_chance else self.base_color
                column.append((char, style))
            # Pad with spaces to height
            column.extend([(' ', self.base_color)] * (self.height - len(column)))
            random.shuffle(column) # Mix them up initially

            self.columns.append(column)
            self.column_speeds.append(random.randint(1, 5)) # Update every 1-5 frames
            self.column_next_update.append(0)


    def update(self) -> Optional[str]:
        self.frame_count +=1 # Manually increment as BaseEffect's update is overridden

        # Update columns that are due
        for col_idx in range(self.width):
            if self.frame_count >= self.column_next_update[col_idx]:
                # Shift column down
                last_char_tuple = self.columns[col_idx].pop()

                # New char at top
                new_char = random.choice(self.all_chars)
                new_style = self.highlight_color if new_char in self.highlight_chars or random.random() < self.highlight_chance else self.base_color
                self.columns[col_idx].insert(0, (new_char, new_style))

                self.column_next_update[col_idx] = self.frame_count + self.column_speeds[col_idx]

        # Prepare render grid (list of lists of styled characters)
        render_grid: List[List[str]] = [] # Each string is already Rich-escaped and styled

        for r_idx in range(self.height):
            line_segments = []
            for c_idx in range(self.width):
                char, style = self.columns[c_idx][r_idx]
                escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                line_segments.append(f"[{style}]{escaped_char}[/{style}]")
            render_grid.append(line_segments)

        # Title and Subtitle Reveal (similar to MatrixRainEffect)
        elapsed = time.time() - self.start_time
        if elapsed > 0.5: # Start revealing title
            self.title_reveal_progress = min(1.0, (elapsed - 0.5) / 1.5) # Slower reveal
            title_len_to_show = int(len(self.title) * self.title_reveal_progress)

            title_row = self.height // 2 - 2
            title_start_col = (self.width - len(self.title)) // 2

            if 0 <= title_row < self.height:
                for i in range(len(self.title)):
                    if title_start_col + i < self.width:
                        if i < title_len_to_show:
                            char_to_draw = self.title[i].replace('[', r'\[')
                            render_grid[title_row][title_start_col + i] = f"[{self.title_style}]{char_to_draw}[/{self.title_style}]"
                        else: # Keep rain char but make it almost invisible or a background color
                             render_grid[title_row][title_start_col + i] = "[on black] [/on black]"


        if elapsed > 1.0: # Start revealing subtitle
            self.subtitle_reveal_progress = min(1.0, (elapsed - 1.0) / 1.5)
            subtitle_len_to_show = int(len(self.subtitle) * self.subtitle_reveal_progress)

            subtitle_row = self.height // 2
            subtitle_start_col = (self.width - len(self.subtitle)) // 2

            if 0 <= subtitle_row < self.height:
                for i in range(len(self.subtitle)):
                     if subtitle_start_col + i < self.width:
                        if i < subtitle_len_to_show:
                            char_to_draw = self.subtitle[i].replace('[', r'\[')
                            render_grid[subtitle_row][subtitle_start_col + i] = f"[{self.subtitle_style}]{char_to_draw}[/{self.subtitle_style}]"
                        else:
                            render_grid[subtitle_row][subtitle_start_col + i] = "[on black] [/on black]"

        final_lines = ["".join(line_segments) for line_segments in render_grid]
        return "\n".join(final_lines)


class LoadingBarEffect(BaseEffect):
    """Displays an ASCII loading bar that fills based on SplashScreen's progress."""

    def __init__(
        self,
        parent_widget: Any, # This will be the SplashScreen instance
        bar_frame_content: str, # ASCII for the empty bar e.g., "[----------]"
        fill_char: str = "#",
        bar_style: str = "bold green",
        text_above: str = "LOADING MODULES...",
        text_below: str = "{progress:.0f}% Complete", # Format string for progress
        text_style: str = "white",
        width: int = 80, # Target width for centering
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.bar_frame_content = bar_frame_content.strip()
        self.fill_char = fill_char[0] if fill_char else "#" # Ensure single char
        self.bar_style = bar_style
        self.text_above = text_above
        self.text_below_template = text_below
        self.text_style = text_style
        self.width = width

        # Try to determine bar width from frame_content (excluding brackets/ends)
        self.bar_interior_width = len(self.bar_frame_content) - 2 # Assuming frame is like [---]
        if self.bar_interior_width <= 0:
            self.bar_interior_width = 20 # Default if frame is unusual

    def update(self) -> Optional[str]:
        # Access progress from the parent SplashScreen widget
        # The parent SplashScreen widget has a reactive 'progress' attribute (0.0 to 1.0)
        current_progress = self.parent.progress if hasattr(self.parent, 'progress') else 0.0

        num_filled = int(current_progress * self.bar_interior_width)
        num_empty = self.bar_interior_width - num_filled

        # Construct the bar
        # Assuming bar_frame_content is like "[--------------------]"
        bar_start = self.bar_frame_content[0]
        bar_end = self.bar_frame_content[-1]

        filled_part = self.fill_char * num_filled
        empty_part = self.bar_frame_content[1+num_filled : 1+num_filled+num_empty] # Get actual empty chars from frame

        # Ensure the bar is always the correct total interior width
        current_bar_interior = filled_part + empty_part
        if len(current_bar_interior) < self.bar_interior_width:
            current_bar_interior += self.bar_frame_content[1+len(current_bar_interior)] * (self.bar_interior_width - len(current_bar_interior))
        elif len(current_bar_interior) > self.bar_interior_width:
            current_bar_interior = current_bar_interior[:self.bar_interior_width]


        styled_bar = f"[{self.bar_style}]{bar_start}{current_bar_interior}{bar_end}[/{self.bar_style}]"

        # Format text below with current progress
        # The parent also has 'progress_text' which might be more descriptive
        progress_percentage_text = self.parent.progress_text if hasattr(self.parent, 'progress_text') and self.parent.progress_text else ""

        # Use the template if available, otherwise use the SplashScreen's progress_text
        if "{progress}" in self.text_below_template:
             text_below_formatted = self.text_below_template.format(progress=current_progress * 100)
        else:
            text_below_formatted = progress_percentage_text if progress_percentage_text else f"{current_progress*100:.0f}%"


        # Centering text and bar (approximate)
        output_lines = []
        if self.text_above:
            pad_above = (self.width - len(self.text_above)) // 2
            escaped_text = self.text_above.replace('[', ESCAPED_OPEN_BRACKET)
            output_lines.append(f"[{self.text_style}]{' ' * pad_above}{escaped_text}{' ' * pad_above}[/{self.text_style}]")
        else:
            output_lines.append("") # Keep spacing consistent

        pad_bar = (self.width - len(self.bar_frame_content)) // 2
        output_lines.append(f"{' ' * pad_bar}{styled_bar}")

        if text_below_formatted:
            pad_below = (self.width - len(text_below_formatted)) // 2
            escaped_text_below = text_below_formatted.replace('[', ESCAPED_OPEN_BRACKET)
            output_lines.append(f"[{self.text_style}]{' ' * pad_below}{escaped_text_below}{' ' * pad_below}[/{self.text_style}]")
        else:
            output_lines.append("")

        # Add some blank lines for spacing if needed, assuming height of ~5-7 lines for this effect
        while len(output_lines) < 5: # Assuming a small vertical footprint
            output_lines.insert(0, "") # Add blank lines at the top for centering
            if len(output_lines) >=5: break
            output_lines.append("") # Add blank lines at the bottom

        return '\n'.join(output_lines[:self.parent.height if hasattr(self.parent, 'height') else 7])


class StarfieldEffect(BaseEffect):
    """Simulates a starfield warp effect."""

    @dataclass
    class Star:
        x: float # Current screen x
        y: float # Current screen y
        z: float # Depth (distance from viewer, max_depth is furthest)
        # For warp effect, stars also need a fixed trajectory from center
        angle: float # Angle of trajectory from center
        initial_speed_factor: float # Base speed factor for this star

    def __init__(
        self,
        parent_widget: Any,
        title: str = "WARP SPEED ENGAGED",
        num_stars: int = 150,
        warp_factor: float = 0.2, # Controls how fast z decreases and thus apparent speed
        max_depth: float = 50.0, # Furthest z value
        star_chars: List[str] = list("·.*+"), # Smallest to largest/brightest
        star_styles: List[str] = ["dim white", "white", "bold white", "bold yellow"],
        width: int = 80,
        height: int = 24,
        title_style: str = "bold cyan on black",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.num_stars = num_stars
        self.warp_factor = warp_factor
        self.max_depth = max_depth
        self.star_chars = star_chars
        self.star_styles = star_styles
        self.width = width
        self.height = height
        self.center_x = width / 2.0
        self.center_y = height / 2.0
        self.title_style = title_style
        self.stars: List[StarfieldEffect.Star] = []
        for _ in range(self.num_stars):
            self.stars.append(self._spawn_star(is_initial_spawn=True))

    def _spawn_star(self, is_initial_spawn: bool = False) -> Star:
        angle = random.uniform(0, 2 * math.pi)
        initial_speed_factor = random.uniform(0.2, 1.0) # How fast it moves from center

        z = self.max_depth # Always spawn at max depth for this warp effect

        return StarfieldEffect.Star(
            x=self.center_x,
            y=self.center_y,
            z=z,
            angle=angle,
            initial_speed_factor=initial_speed_factor
        )

    def update(self) -> Optional[str]:
        styled_chars_on_grid: Dict[Tuple[int, int], Tuple[str, str]] = {}

        for i in range(len(self.stars)):
            star = self.stars[i]
            star.z -= self.warp_factor

            if star.z <= 0:
                self.stars[i] = self._spawn_star()
                continue

            radius_on_screen = star.initial_speed_factor * (self.max_depth - star.z) * (self.width / (self.max_depth * 10.0))


            star.x = self.center_x + math.cos(star.angle) * radius_on_screen
            # Adjust y movement based on aspect ratio if terminal cells aren't square
            # Assuming roughly 2:1 height:width for characters, so y movement appears slower
            star.y = self.center_y + math.sin(star.angle) * radius_on_screen * 0.5

            z_ratio = star.z / self.max_depth

            char_idx = 0
            if z_ratio < 0.25: char_idx = 3
            elif z_ratio < 0.50: char_idx = 2
            elif z_ratio < 0.75: char_idx = 1
            else: char_idx = 0

            char_idx = min(char_idx, len(self.star_chars) - 1)
            style_idx = min(char_idx, len(self.star_styles) -1)

            star_char = self.star_chars[char_idx]
            star_style = self.star_styles[style_idx]

            ix, iy = int(star.x), int(star.y)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                styled_chars_on_grid[(ix, iy)] = (star_char, star_style)

        output_lines = []
        for r_idx in range(self.height):
            line_segments = []
            for c_idx in range(self.width):
                if (c_idx, r_idx) in styled_chars_on_grid:
                    char, style = styled_chars_on_grid[(c_idx, r_idx)]
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(' ')
            output_lines.append("".join(line_segments))

        if self.title:
            title_y = self.height // 2
            title_x_start = (self.width - len(self.title)) // 2

            if 0 <= title_y < self.height:
                title_segments = []
                current_title_char_idx = 0
                for c_idx in range(self.width):
                    is_title_char = title_x_start <= c_idx < title_x_start + len(self.title)
                    if is_title_char:
                        char_to_draw = self.title[current_title_char_idx].replace('[', r'\[')
                        title_segments.append(f"[{self.title_style}]{char_to_draw}[/{self.title_style}]")
                        current_title_char_idx +=1
                    else:
                        if (c_idx, title_y) in styled_chars_on_grid: # Star is behind title char
                            char, style = styled_chars_on_grid[(c_idx, title_y)]
                            escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                            title_segments.append(f"[{style}]{escaped_char}[/{style}]")
                        else: # Empty space behind title char
                            title_segments.append(' ')
                output_lines[title_y] = "".join(title_segments)

        return "\n".join(output_lines)


class SpotlightEffect(BaseEffect):
    """Moves a 'spotlight' over content, revealing parts of it."""

    def __init__(
        self,
        parent_widget: Any,
        background_content: str,
        spotlight_radius: int = 5,
        movement_speed: float = 10.0, # Pixels (grid cells) per second
        path_type: str = "lissajous", # "lissajous", "random_walk", "circle"
        visible_style: str = "bold white", # Style of text inside spotlight
        hidden_style: str = "dim black on black", # Style of text outside spotlight (very dim)
        width: int = 80, # display width
        height: int = 24, # display height
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.lines = background_content.splitlines()
        # Normalize background content dimensions
        self.content_height = len(self.lines)
        self.content_width = max(len(line) for line in self.lines) if self.lines else 0

        self.padded_lines = []
        for i in range(self.content_height):
            line = self.lines[i]
            self.padded_lines.append(line + ' ' * (self.content_width - len(line)))

        self.spotlight_radius = spotlight_radius
        self.spotlight_radius_sq = spotlight_radius ** 2
        self.movement_speed = movement_speed # Cells per second
        self.path_type = path_type
        self.visible_style = visible_style
        self.hidden_style = hidden_style
        self.display_width = width # Max width of the rendered output
        self.display_height = height # Max height

        self.spotlight_x = float(self.content_width // 2)
        self.spotlight_y = float(self.content_height // 2)

        # Path specific parameters
        self.time_elapsed_for_path = 0
        if self.path_type == "random_walk":
            self.vx = random.uniform(-self.movement_speed, self.movement_speed)
            self.vy = random.uniform(-self.movement_speed, self.movement_speed)

        self.time_at_last_frame = time.time()


    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame
        self.time_at_last_frame = current_time
        self.time_elapsed_for_path += delta_time

        # Update spotlight position
        if self.path_type == "lissajous":
            # Simple Lissajous curve for movement
            # Adjust frequencies (e.g., 0.1, 0.07) and phase for different paths
            self.spotlight_x = (self.content_width / 2) + (self.content_width / 2 - self.spotlight_radius) * math.sin(self.time_elapsed_for_path * 0.15)
            self.spotlight_y = (self.content_height / 2) + (self.content_height / 2 - self.spotlight_radius) * math.cos(self.time_elapsed_for_path * 0.1)
        elif self.path_type == "circle":
            radius = min(self.content_width, self.content_height) / 2 - self.spotlight_radius
            self.spotlight_x = (self.content_width / 2) + radius * math.cos(self.time_elapsed_for_path * self.movement_speed * 0.02) # speed factor
            self.spotlight_y = (self.content_height / 2) + radius * math.sin(self.time_elapsed_for_path * self.movement_speed * 0.02)
        elif self.path_type == "random_walk":
            self.spotlight_x += self.vx * delta_time
            self.spotlight_y += self.vy * delta_time
            # Boundary checks and bounce / change direction
            if not (0 <= self.spotlight_x < self.content_width):
                self.vx *= -1
                self.spotlight_x = max(0, min(self.spotlight_x, self.content_width -1))
            if not (0 <= self.spotlight_y < self.content_height):
                self.vy *= -1
                self.spotlight_y = max(0, min(self.spotlight_y, self.content_height -1))
            # Occasionally change direction randomly
            if random.random() < 0.01: # 1% chance per frame
                 self.vx = random.uniform(-self.movement_speed, self.movement_speed) * 0.1 # Slower random walk
                 self.vy = random.uniform(-self.movement_speed, self.movement_speed) * 0.1


        # Render the content with spotlight effect
        output_lines = []
        # Determine rendering bounds based on display_height/width and content_height/width
        render_height = min(self.display_height, self.content_height)

        # Center the content if smaller than display area
        content_start_row = (self.display_height - render_height) // 2

        for r_disp in range(self.display_height):
            if content_start_row <= r_disp < content_start_row + render_height:
                r_content = r_disp - content_start_row # Index in self.padded_lines
                line = self.padded_lines[r_content]
                styled_line_segments = []

                content_start_col = (self.display_width - self.content_width) // 2

                for c_disp in range(self.display_width):
                    if content_start_col <= c_disp < content_start_col + self.content_width:
                        c_content = c_disp - content_start_col # Index in line
                        char = line[c_content]
                        escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)

                        # Check distance from spotlight center
                        # Adjust for character aspect ratio if desired (y distances count more)
                        dist_sq = (c_content - self.spotlight_x)**2 + ((r_content - self.spotlight_y)*2)**2 # Y weighted

                        if dist_sq <= self.spotlight_radius_sq:
                            styled_line_segments.append(f"[{self.visible_style}]{escaped_char}[/{self.visible_style}]")
                        else:
                            # Optional: fade effect at edges of spotlight
                            # For now, simple binary visible/hidden
                            styled_line_segments.append(f"[{self.hidden_style}]{escaped_char}[/{self.hidden_style}]")
                    else: # Outside content width, but within display width (padding)
                        styled_line_segments.append(f"[{self.hidden_style}] [/{self.hidden_style}]")
                output_lines.append("".join(styled_line_segments))
            else: # Outside content height (padding)
                output_lines.append(f"[{self.hidden_style}]{' ' * self.display_width}[/{self.hidden_style}]")

        return "\n".join(output_lines)


class SoundBarsEffect(BaseEffect):
    """Simulates abstract sound visualizer bars."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Audio Core Calibrating...",
        num_bars: int = 15,
        max_bar_height: Optional[int] = None, # If None, calculated from display_height
        bar_char_filled: str = "█",
        bar_char_empty: str = " ", # Usually not visible if styled with background
        bar_styles: List[str] = ["bold blue", "bold magenta", "bold cyan", "bold green", "bold yellow", "bold red"],
        width: int = 80, # display width
        height: int = 24, # display height
        title_style: str = "bold white",
        update_speed: float = 0.05, # How fast bars change height
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.num_bars = num_bars
        self.display_width = width
        self.display_height = height

        # Title takes ~1 line + 1 for spacing, rest for bars
        self.title_area_height = 2 if self.title else 0
        self.max_bar_height = max_bar_height if max_bar_height is not None else self.display_height - self.title_area_height -1 # -1 for base line
        if self.max_bar_height <=0: self.max_bar_height = 1

        self.bar_char_filled = bar_char_filled[0]
        self.bar_char_empty = bar_char_empty[0]
        self.bar_styles = bar_styles
        self.title_style = title_style
        self.update_speed = update_speed # Interval for changing bar heights

        self.bar_heights = [random.randint(1, self.max_bar_height) for _ in range(self.num_bars)]
        self.bar_targets = list(self.bar_heights) # Target heights for smooth transition
        self.bar_colors = [random.choice(self.bar_styles) for _ in range(self.num_bars)]

        self._last_bar_update_time = time.time()

    def _update_bar_heights(self):
        """Update target heights and smoothly move current heights."""
        for i in range(self.num_bars):
            # Chance to pick a new target height
            if random.random() < 0.2 or self.bar_heights[i] == self.bar_targets[i]:
                self.bar_targets[i] = random.randint(1, self.max_bar_height)
                self.bar_colors[i] = random.choice(self.bar_styles) # Change color too
            # Move towards target
            if self.bar_heights[i] < self.bar_targets[i]:
                self.bar_heights[i] = min(self.bar_targets[i], self.bar_heights[i] + 1) # Step of 1 for simplicity
            elif self.bar_heights[i] > self.bar_targets[i]:
                 self.bar_heights[i] = max(self.bar_targets[i], self.bar_heights[i] -1)


    def update(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self._last_bar_update_time >= self.update_speed:
            self._update_bar_heights()
            self._last_bar_update_time = current_time

        output_lines = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styled_output_lines = [""] * self.display_height

        # Render title if present
        title_start_row = 0
        if self.title:
            title_x_start = (self.display_width - len(self.title)) // 2
            for c, char_val in enumerate(self.title):
                if title_x_start + c < self.display_width:
                    output_lines[title_start_row][title_x_start + c] = (char_val.replace('[',r'\['), self.title_style)
            title_start_row += 1 # Move down for next line (e.g. spacing or bars)
            if self.title_area_height > 1: # if spacing was reserved
                 title_start_row += (self.title_area_height -1)


        # Render bars
        # Calculate bar width and spacing (simple equal spacing)
        bar_display_area_width = self.display_width
        total_bar_chars_width = self.num_bars # Assuming each bar is 1 char wide

        # If we want wider bars, e.g. 2 chars wide:
        # bar_char_width = 2
        # total_bar_chars_width = self.num_bars * bar_char_width
        # For simplicity, 1 char wide bars.

        spacing = (bar_display_area_width - total_bar_chars_width) // (self.num_bars + 1)
        if spacing < 0: spacing = 0 # Bars might overlap if too many

        current_c = spacing # Start position for first bar

        for i in range(self.num_bars):
            if current_c >= self.display_width: break # No more space for bars

            bar_h = self.bar_heights[i]
            bar_style_to_use = self.bar_colors[i]

            for r in range(self.max_bar_height):
                # Bars are drawn from bottom up
                row_idx_on_display = (title_start_row + self.max_bar_height -1) - r
                if row_idx_on_display < title_start_row : continue # Don't draw into title area from bottom
                if row_idx_on_display >= self.display_height : continue


                if r < bar_h : # This part of bar is filled
                    output_lines[row_idx_on_display][current_c] = (self.bar_char_filled, bar_style_to_use)
                else: # This part is empty (above the current bar height)
                    output_lines[row_idx_on_display][current_c] = (self.bar_char_empty, "default") # or specific empty style

            current_c += 1 # Next char column for this bar (if multi-char wide)
            current_c += spacing # Move to start of next bar

        # Convert the character grid to styled lines
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                cell = output_lines[r_idx][c_idx]
                if isinstance(cell, tuple):
                    char, style = cell
                    line_segments.append(f"[{style}]{char}[/{style}]")
                else: # Space
                    line_segments.append(' ') # Default background
            styled_output_lines[r_idx] = "".join(line_segments)

        return "\n".join(styled_output_lines)


class MazeGeneratorEffect(BaseEffect):
    """Animates the generation of a random maze using Depth First Search."""

    CELL_PATH_N = 1
    CELL_PATH_E = 2
    CELL_PATH_S = 4
    CELL_PATH_W = 8
    CELL_VISITED = 16

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Generating Labyrinth...",
        maze_width: int = 39, # Grid cells (must be odd for typical wall representation)
        maze_height: int = 19, # Grid cells (must be odd)
        wall_char: str = "█",
        path_char: str = " ",
        cursor_char: str = "░", # Char for the current cell being processed
        wall_style: str = "bold blue",
        path_style: str = "on black", # Path is often just background
        cursor_style: str = "bold yellow",
        title_style: str = "bold white",
        generation_speed: float = 0.01, # Delay between steps of generation
        display_width: int = 80, # Total splash screen width
        display_height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        # Ensure maze dimensions are odd for typical cell/wall structure
        self.maze_cols = maze_width if maze_width % 2 != 0 else maze_width -1
        self.maze_rows = maze_height if maze_height % 2 != 0 else maze_height -1
        if self.maze_cols < 3: self.maze_cols = 3
        if self.maze_rows < 3: self.maze_rows = 3

        self.wall_char = wall_char
        self.path_char = path_char
        self.cursor_char = cursor_char
        self.wall_style = wall_style
        self.path_style = path_style
        self.cursor_style = cursor_style
        self.title_style = title_style
        self.generation_speed = generation_speed # Interval for maze generation steps
        self.display_width = display_width
        self.display_height = display_height

        # Maze grid: stores bitmasks for paths and visited status
        self.maze_grid = [[0 for _ in range(self.maze_cols)] for _ in range(self.maze_rows)]
        self.stack = [] # For DFS algorithm

        # Start DFS from a random cell (must be an actual cell, not a wall position)
        # In our grid, (0,0) is a cell.
        self.current_cx = random.randrange(0, self.maze_cols, 2) # Ensure starting on a "cell" column if we consider walls
        self.current_cy = random.randrange(0, self.maze_rows, 2) # Ensure starting on a "cell" row
        # Simpler: map to a conceptual grid of cells (width/2, height/2) then map back to drawing grid
        # Let's use a cell-based grid for logic, then render to character grid.
        self.logic_cols = (self.maze_cols +1) // 2
        self.logic_rows = (self.maze_rows +1) // 2
        self.logic_grid = [[0 for _ in range(self.logic_cols)] for _ in range(self.logic_rows)]

        self.current_logic_x = self.current_cx // 2
        self.current_logic_y = self.current_cy // 2
        self.logic_grid[self.current_logic_y][self.current_logic_x] = self.CELL_VISITED
        self.stack.append((self.current_logic_x, self.current_logic_y))

        self.is_generating = True
        self._last_gen_step_time = time.time()

    def _generation_step_dfs(self):
        if not self.stack:
            self.is_generating = False
            return

        x, y = self.stack[-1] # Current cell
        self.current_logic_x, self.current_logic_y = x,y # For cursor drawing

        neighbors = []
        # Check North
        if y > 0 and self.logic_grid[y-1][x] & self.CELL_VISITED == 0: neighbors.append(('N', x, y-1))
        # Check East
        if x < self.logic_cols - 1 and self.logic_grid[y][x+1] & self.CELL_VISITED == 0: neighbors.append(('E', x+1, y))
        # Check South
        if y < self.logic_rows - 1 and self.logic_grid[y+1][x] & self.CELL_VISITED == 0: neighbors.append(('S', x, y+1))
        # Check West
        if x > 0 and self.logic_grid[y][x-1] & self.CELL_VISITED == 0: neighbors.append(('W', x-1, y))

        if neighbors:
            direction, nx, ny = random.choice(neighbors)
            if direction == 'N':
                self.logic_grid[y][x] |= self.CELL_PATH_N
                self.logic_grid[ny][nx] |= self.CELL_PATH_S
            elif direction == 'E':
                self.logic_grid[y][x] |= self.CELL_PATH_E
                self.logic_grid[ny][nx] |= self.CELL_PATH_W
            elif direction == 'S':
                self.logic_grid[y][x] |= self.CELL_PATH_S
                self.logic_grid[ny][nx] |= self.CELL_PATH_N
            elif direction == 'W':
                self.logic_grid[y][x] |= self.CELL_PATH_W
                self.logic_grid[ny][nx] |= self.CELL_PATH_E

            self.logic_grid[ny][nx] |= self.CELL_VISITED
            self.stack.append((nx, ny))
        else:
            self.stack.pop() # Backtrack

    def update(self) -> Optional[str]:
        current_time = time.time()
        if self.is_generating and (current_time - self._last_gen_step_time >= self.generation_speed):
            self._generation_step_dfs()
            self._last_gen_step_time = current_time

        # Render the maze to the display grid (display_width x display_height)
        # The maze itself is self.maze_cols x self.maze_rows characters
        output_grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styled_output_lines = [""] * self.display_height

        maze_start_row = (self.display_height - self.maze_rows) // 2
        maze_start_col = (self.display_width - self.maze_cols) // 2

        for r_draw in range(self.maze_rows):
            for c_draw in range(self.maze_cols):
                r_disp = maze_start_row + r_draw
                c_disp = maze_start_col + c_draw

                if not (0 <= r_disp < self.display_height and 0 <= c_disp < self.display_width):
                    continue # Skip drawing outside display boundary

                char_to_draw = self.wall_char
                style_to_use = self.wall_style

                # Convert draw coords to logic grid cell coords and wall/path determination
                logic_x, logic_y = c_draw // 2, r_draw // 2

                is_wall_row = r_draw % 2 == 1
                is_wall_col = c_draw % 2 == 1

                if not is_wall_row and not is_wall_col: # Cell center
                    char_to_draw = self.path_char
                    style_to_use = self.path_style
                    if self.is_generating and logic_x == self.current_logic_x and logic_y == self.current_logic_y:
                        char_to_draw = self.cursor_char
                        style_to_use = self.cursor_style

                elif not is_wall_row and is_wall_col: # Horizontal wall/path between cells (y, x) and (y, x+1)
                    if logic_x < self.logic_cols -1 and \
                       (self.logic_grid[logic_y][logic_x] & self.CELL_PATH_E or \
                        self.logic_grid[logic_y][logic_x+1] & self.CELL_PATH_W):
                        char_to_draw = self.path_char
                        style_to_use = self.path_style

                elif is_wall_row and not is_wall_col: # Vertical wall/path between cells (y,x) and (y+1,x)
                     if logic_y < self.logic_rows -1 and \
                       (self.logic_grid[logic_y][logic_x] & self.CELL_PATH_S or \
                        self.logic_grid[logic_y+1][logic_x] & self.CELL_PATH_N):
                        char_to_draw = self.path_char
                        style_to_use = self.path_style
                # else it's a wall intersection, keep wall_char

                output_grid[r_disp][c_disp] = (char_to_draw, style_to_use)

        # Convert to styled lines
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                cell = output_grid[r_idx][c_idx]
                if isinstance(cell, tuple):
                    char, style = cell
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else: # Space, apply default path style or background
                    line_segments.append(f"[{self.path_style}] [/{self.path_style}]")
            styled_output_lines[r_idx] = "".join(line_segments)

        # Overlay title
        if self.title:
            title_y = maze_start_row - 2 if maze_start_row > 1 else self.display_height - 1
            if not self.is_generating: title_y = self.display_height // 2 # Center title when done

            title_x_start = (self.display_width - len(self.title)) // 2
            if 0 <= title_y < self.display_height:
                # Simplified title overlay for now: assumes it replaces the line
                title_line_str = self.title.center(self.display_width).replace('[',r'\[')
                styled_output_lines[title_y] = f"[{self.title_style}]{title_line_str}[/{self.title_style}]"

        return "\n".join(styled_output_lines)


class TerminalBootEffect(BaseEffect):
    """Simulates a classic terminal boot-up sequence."""

    @dataclass
    class BootLine:
        text: str
        delay_before: float = 0.1 # Delay before this line starts typing
        type_speed: float = 0.03 # Seconds per character
        pause_after: float = 0.2 # Pause after line is fully typed
        style: str = "default"

    def __init__(
        self,
        parent_widget: Any,
        boot_sequence: List[Dict[str, Any]], # List of dicts for BootLine properties
        width: int = 80,
        height: int = 24,
        cursor: str = "_",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)

        self.lines_to_display: List[TerminalBootEffect.BootLine] = []
        for item in boot_sequence:
            self.lines_to_display.append(TerminalBootEffect.BootLine(**item))

        self.width = width
        self.height = height
        self.cursor_char = cursor

        self.current_line_index = 0
        self.current_char_index = 0
        self.time_since_last_char = 0
        self.time_waiting_for_next_line = 0
        self.time_paused_after_line = 0

        self.output_buffer: List[str] = [""] * self.height # Stores fully typed lines
        self.current_display_line = 0 # Which line in output_buffer we are writing to

        self.state = "delay_before_line" # States: delay_before_line, typing, paused_after_line, done

    def update(self) -> Optional[str]:
        if self.current_line_index >= len(self.lines_to_display):
            self.state = "done"

        elapsed_frame_time = time.time() - self.start_time # More like delta_time if called frequently
        self.start_time = time.time() # Reset for next frame's delta_time calculation

        current_boot_line = None
        if self.state != "done":
            current_boot_line = self.lines_to_display[self.current_line_index]

        if self.state == "delay_before_line":
            self.time_waiting_for_next_line += elapsed_frame_time
            if self.time_waiting_for_next_line >= current_boot_line.delay_before:
                self.state = "typing"
                self.time_waiting_for_next_line = 0

        if self.state == "typing":
            self.time_since_last_char += elapsed_frame_time
            if self.time_since_last_char >= current_boot_line.type_speed:
                self.time_since_last_char = 0

                if self.current_char_index < len(current_boot_line.text):
                    # Add char to current line in buffer
                    # Ensure current_display_line is within height
                    if self.current_display_line >= self.height:
                        # Scroll up output_buffer
                        self.output_buffer.pop(0)
                        self.output_buffer.append("")
                        self.current_display_line = self.height -1

                    self.output_buffer[self.current_display_line] += current_boot_line.text[self.current_char_index]
                    self.current_char_index += 1

                if self.current_char_index >= len(current_boot_line.text):
                    self.state = "paused_after_line"
                    # Move to next line in buffer for next boot message
                    self.current_display_line +=1


        if self.state == "paused_after_line":
            self.time_paused_after_line += elapsed_frame_time
            if self.time_paused_after_line >= current_boot_line.pause_after:
                self.current_line_index += 1
                self.current_char_index = 0
                self.time_paused_after_line = 0
                if self.current_line_index < len(self.lines_to_display):
                    self.state = "delay_before_line"
                else:
                    self.state = "done"

        # Render the output buffer
        final_styled_lines = []

        # To correctly apply styles, we need to track which BootLine generated which line in output_buffer.
        # This is tricky if lines scroll. A simpler approach for now:
        # output_buffer stores plain text. Styles are applied during rendering here.
        # We need a mapping from output_buffer line index to original BootLine style.
        # This is difficult because of scrolling.

        # Simpler styling logic:
        # 1. Completed lines: Use the style from their original BootLine object if we can track it.
        #    If not, use a default.
        # 2. Actively typing line: Use its BootLine style + cursor.
        # 3. Future lines (not yet in buffer): Blank.

        # Let's assume output_buffer[i] was generated by lines_to_display[k]
        # where k is what current_line_index was when output_buffer[i] was completed.
        # This requires storing style info alongside text in output_buffer or a parallel structure.

        # Simplification for this version:
        # All fully typed lines will use their defined style.
        # The line currently being typed shows cursor.

        # Store (text, style_name) in output_buffer
        # Modify how output_buffer is populated:
        # When a line is completed (state becomes "paused_after_line"):
        #   self.output_buffer[self.current_display_line-1] = (text, style)
        # When a character is typed:
        #   self.output_buffer[self.current_display_line] = (current_text_so_far, style_of_current_line)

        # For this pass, I will keep output_buffer as list of strings and try to reconstruct styles.
        # This is not ideal but avoids changing the buffer structure significantly now.

        # Determine which original boot lines are visible or partially visible due to scrolling
        # This logic gets very complex with scrolling.
        # Let's assume no scrolling for v1 of this effect to simplify styling.
        # If current_display_line >= height, it means we should have scrolled.
        # The current code already handles scrolling of output_buffer.

        # Let's find the style for each line in the buffer.
        # The line `self.output_buffer[j]` corresponds to `self.lines_to_display[self.current_line_index - (self.current_display_line - j)]`
        # if we assume a direct mapping without considering complex scrolling scenarios.

        for i in range(self.height):
            line_text = self.output_buffer[i] if i < len(self.output_buffer) else ""
            line_to_render = line_text.replace('[', r'\[')
            style_to_use = "default" # Default style

            # Try to find the original BootLine that corresponds to this output_buffer line
            # This is an approximation assuming lines map somewhat directly if no/simple scrolling.
            # `boot_line_source_index` is the estimated index in `self.lines_to_display`
            # This logic is tricky because `current_display_line` increments AFTER a line is full.

            # If line `i` is the one being currently typed or was just finished:
            is_active_typing_line = (self.state == "typing" and i == self.current_display_line)

            # Determine the source BootLine for styling
            # This needs to account for scrolling. If output_buffer[0] was line k,
            # output_buffer[1] was line k+1, etc., until scroll.
            # This is hard to backtrack perfectly without more info.
            # Simplified: Use current_boot_line's style for active line, default for others.
            # This won't style completed lines with their original styles correctly if styles vary.

            # A better simple approach: store the style with the text in output_buffer.
            # Let's assume self.output_buffer stores tuples: (text, style_str)
            # This requires changing how self.output_buffer is populated.
            # For now, I'll stick to the current simpler (but less accurate styling) model.

            if is_active_typing_line and current_boot_line:
                style_to_use = current_boot_line.style
                final_styled_lines.append(f"[{style_to_use}]{line_to_render}{self.cursor_char}[/{style_to_use}]")
            elif self.state == "paused_after_line" and i == self.current_display_line -1 and current_boot_line:
                # Line just finished, use its style, no cursor yet for next line
                # current_boot_line here is the one that just finished.
                 style_to_use = current_boot_line.style
                 final_styled_lines.append(f"[{style_to_use}]{line_to_render}[/{style_to_use}]")
            elif self.state == "done" and i == self.current_display_line -1 and self.current_display_line > 0:
                 # Last line that was typed.
                 if self.lines_to_display:
                    last_typed_line_style = self.lines_to_display[len(self.lines_to_display)-1].style
                    final_styled_lines.append(f"[{last_typed_line_style}]{line_to_render}[/{last_typed_line_style}]")
                 else:
                    final_styled_lines.append(f"[default]{line_to_render}[/default]")
            else:
                # For other lines (already scrolled or not yet typed fully on screen part)
                # This is where it's hard to get the original style back easily.
                # Use default for lines that are "old" or if style cannot be determined.
                # If line_text is not empty, it means it was typed.
                if line_text: # It's a previously completed line
                    # Try to guess its original style - this is the weak point.
                    # For now, let's assume if it has text, it's from a previous line.
                    # This part needs a more robust solution for varied styles on completed lines.
                    # Simplest: use default for all non-active lines.
                     final_styled_lines.append(f"[default]{line_to_render}[/default]")
                else:
                     final_styled_lines.append(line_to_render) # Empty line

        return "\n".join(final_styled_lines)


class GlitchRevealEffect(BaseEffect):
    """Reveals content by starting glitchy and becoming clear over time."""

    def __init__(
        self,
        parent_widget: Any,
        content: str, # The clear, final content
        duration: float = 2.0, # Total duration of the reveal effect
        glitch_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?",
        start_intensity: float = 0.8, # Initial glitch intensity (0.0 to 1.0)
        end_intensity: float = 0.0,   # Final glitch intensity
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.clear_content = content
        self.duration = duration
        self.glitch_chars = glitch_chars
        self.start_intensity = start_intensity
        self.end_intensity = end_intensity

        self.lines = self.clear_content.strip().split('\n')

    def update(self) -> Optional[str]:
        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration) # Normalized time (0 to 1)

        # Intensity decreases over time (linear interpolation)
        current_intensity = self.start_intensity + (self.end_intensity - self.start_intensity) * progress
        # Could use an easing function for non-linear change in intensity.

        if current_intensity <= 0.01: # Effectively clear
            return self.clear_content.replace('[',r'\[')


        glitched_lines = []
        for line_idx, line_text in enumerate(self.lines):
            glitched_line_chars = list(line_text)
            for char_idx in range(len(glitched_line_chars)):
                if random.random() < current_intensity:
                    # Chance to replace char, or shift it, or change color
                    if random.random() < 0.7: # Replace char
                        glitched_line_chars[char_idx] = random.choice(self.glitch_chars)
                    # Could add other glitch types like small offsets or color shifts here
            glitched_lines.append("".join(glitched_line_chars))

        # Basic styling for glitched parts - can be enhanced
        output_lines = []
        for line in glitched_lines:
            escaped_line = line.replace('[', r'\[')
            # Randomly apply a "glitchy" color style to some parts
            if random.random() < current_intensity * 0.5: # More styling when more intense
                style = random.choice(["bold red", "bold blue", "bold yellow", "bold magenta"])
                output_lines.append(f"[{style}]{escaped_line}[/{style}]")
            else:
                output_lines.append(escaped_line) # Rely on card's base style

        return "\n".join(output_lines)


class AsciiMorphEffect(BaseEffect):
    """Smoothly morphs one ASCII art into another."""

    def __init__(
        self,
        parent_widget: Any,
        start_content: str,
        end_content: str,
        duration: float = 2.0, # Total duration of the morph
        morph_style: str = "dissolve", # "dissolve", "random_pixel", "wipe_left_to_right"
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.start_lines = start_content.splitlines()
        self.end_lines = end_content.splitlines()
        self.duration = duration
        self.morph_style = morph_style

        # Normalize line lengths and line counts for consistent morphing
        self.height = max(len(self.start_lines), len(self.end_lines))
        self.width = 0
        for line in self.start_lines + self.end_lines:
            if len(line) > self.width:
                self.width = len(line)

        self.start_lines = self._pad_art(self.start_lines)
        self.end_lines = self._pad_art(self.end_lines)

        # For 'dissolve' or 'random_pixel', precompute all character positions
        self.all_positions = []
        if self.morph_style in ["dissolve", "random_pixel"]:
            for r in range(self.height):
                for c in range(self.width):
                    if self.start_lines[r][c] != self.end_lines[r][c]:
                        self.all_positions.append((r, c))
            if self.morph_style == "dissolve": # Shuffle for dissolve
                random.shuffle(self.all_positions)

        self.current_art_chars = [list(line) for line in self.start_lines]

    def _pad_art(self, art_lines: List[str]) -> List[str]:
        """Pads ASCII art to consistent width and height."""
        padded_art = []
        for i in range(self.height):
            if i < len(art_lines):
                line = art_lines[i]
                padded_art.append(line + ' ' * (self.width - len(line)))
            else:
                padded_art.append(' ' * self.width)
        return padded_art

    def update(self) -> Optional[str]:
        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration)

        if progress >= 1.0:
            return "\n".join(self.end_lines).replace('[',r'\[')

        if self.morph_style == "dissolve" or self.morph_style == "random_pixel":
            num_chars_to_change = int(progress * len(self.all_positions))
            for i in range(num_chars_to_change):
                if i < len(self.all_positions):
                    r, c = self.all_positions[i]
                    if self.morph_style == "dissolve":
                         # For dissolve, directly set to end char
                        self.current_art_chars[r][c] = self.end_lines[r][c]
                    elif self.morph_style == "random_pixel":
                        # For random_pixel, set to a random char during transition, then final
                        # This needs another state or to be driven by progress.
                        # Simpler: if not fully progressed, pick start or end based on sub-progress for that pixel
                        if random.random() < progress: # As progress increases, more chance to be end_char
                           self.current_art_chars[r][c] = self.end_lines[r][c]
                        else:
                           self.current_art_chars[r][c] = self.start_lines[r][c] # Or a random char

            # For random_pixel, we should re-evaluate all pixels each frame based on progress
            if self.morph_style == "random_pixel":
                 for r_idx in range(self.height):
                    for c_idx in range(self.width):
                        if self.start_lines[r_idx][c_idx] != self.end_lines[r_idx][c_idx]:
                            if random.random() < progress:
                                self.current_art_chars[r_idx][c_idx] = self.end_lines[r_idx][c_idx]
                            else:
                                # Optionally, insert a random "transition" character
                                # self.current_art_chars[r_idx][c_idx] = random.choice(".:-=+*#%@")
                                self.current_art_chars[r_idx][c_idx] = self.start_lines[r_idx][c_idx]
                        else:
                            self.current_art_chars[r_idx][c_idx] = self.start_lines[r_idx][c_idx]


        elif self.morph_style == "wipe_left_to_right":
            wipe_column = int(progress * self.width)
            for r in range(self.height):
                for c in range(self.width):
                    if c < wipe_column:
                        self.current_art_chars[r][c] = self.end_lines[r][c]
                    else:
                        self.current_art_chars[r][c] = self.start_lines[r][c]

        # Default or fallback: simple crossfade (alpha blending not possible with chars)
        # So, stick to one of the above, or make dissolve the default.
        # If morph_style is not recognized, it will effectively be stuck on start_art or do random_pixel if all_positions was populated.
        # Let's ensure 'dissolve' is the default if style is unknown.
        else: # Fallback or if morph_style == "dissolve" initially
            num_chars_to_change = int(progress * len(self.all_positions))
            for i in range(num_chars_to_change):
                if i < len(self.all_positions):
                    r, c = self.all_positions[i]
                    self.current_art_chars[r][c] = self.end_lines[r][c]


        return "\n".join("".join(row) for row in self.current_art_chars).replace('[',r'\[')


class RaindropsEffect(BaseEffect):
    """Simulates raindrops creating ripples on a pond surface."""

    @dataclass
    class Ripple:
        cx: int # Center x
        cy: int # Center y
        radius: float = 0.0
        max_radius: int = 5
        current_char_index: int = 0
        speed: float = 1.0 # Radius increase per second
        life: float = 2.0 # Lifespan in seconds
        alive_time: float = 0.0
        style: str = "blue"

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Aqua Reflections",
        width: int = 80,
        height: int = 24,
        spawn_rate: float = 0.5, # Average drops per second
        ripple_chars: List[str] = list("·oO()"), # Smallest to largest, then fades maybe
        ripple_styles: List[str] = ["blue", "cyan", "dim blue"],
        max_concurrent_ripples: int = 15,
        base_water_char: str = "~",
        water_style: str = "dim blue",
        title_style: str = "bold white on blue",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.display_width = width
        self.display_height = height
        self.spawn_rate = spawn_rate
        self.ripple_chars = ripple_chars
        self.ripple_styles = ripple_styles
        self.max_concurrent_ripples = max_concurrent_ripples
        self.base_water_char = base_water_char
        self.water_style = water_style
        self.title_style = title_style

        self.ripples: List[RaindropsEffect.Ripple] = []
        self.time_since_last_spawn = 0.0
        self.time_at_last_frame = time.time()

    def _spawn_ripple(self):
        if len(self.ripples) < self.max_concurrent_ripples:
            cx = random.randint(0, self.display_width -1)
            cy = random.randint(0, self.display_height -1)
            max_r = random.randint(3, 8)
            speed = random.uniform(3.0, 6.0) # Faster ripples
            life = random.uniform(0.8, 1.5)   # Shorter lifespan
            style = random.choice(self.ripple_styles)
            self.ripples.append(RaindropsEffect.Ripple(cx=cx, cy=cy, max_radius=max_r, speed=speed, life=life, style=style))

    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame
        self.time_at_last_frame = current_time

        # Spawn new ripples
        self.time_since_last_spawn += delta_time
        if self.time_since_last_spawn * self.spawn_rate >= 1.0:
            self._spawn_ripple()
            self.time_since_last_spawn = 0.0
            # Could spawn multiple if spawn_rate is high and delta_time was large
            while random.random() < (self.time_since_last_spawn * self.spawn_rate) -1:
                 self._spawn_ripple()
                 self.time_since_last_spawn -= 1.0/self.spawn_rate


        # Update and filter ripples
        active_ripples = []
        for ripple in self.ripples:
            ripple.alive_time += delta_time
            if ripple.alive_time < ripple.life:
                ripple.radius += ripple.speed * delta_time
                # Determine current ripple character based on radius progression
                # Progress through chars as radius grows, then maybe fade
                char_progress = (ripple.radius / ripple.max_radius) * (len(self.ripple_chars) -1)
                ripple.current_char_index = min(len(self.ripple_chars)-1, int(char_progress))
                active_ripples.append(ripple)
        self.ripples = active_ripples

        # Render to a grid first, handling overlaps (newer/smaller ripples on top conceptually)
        # For simplicity, let's not handle complex overlaps perfectly. Last drawn wins.
        # Initialize with base water pattern
        char_grid = [[(self.base_water_char, self.water_style) for _ in range(self.display_width)] for _ in range(self.display_height)]

        for ripple in sorted(self.ripples, key=lambda r: r.radius, reverse=True): # Draw larger (older) ripples first
            char_to_use = self.ripple_chars[ripple.current_char_index]
            # Could also fade style based on ripple.life vs ripple.alive_time

            # Draw the circle (approximate)
            # This is a simple way to draw a circle on a grid. More advanced algorithms exist.
            for angle_deg in range(0, 360, 10): # Draw points on the circle
                angle_rad = math.radians(angle_deg)
                # For character aspect ratio, y movement might need scaling if cells aren't square
                # Assume x_scale = 1, y_scale = 0.5 (chars are twice as tall as wide)
                # So, for a visually circular ripple, the "y radius" in grid cells is smaller.
                # Let's draw actual grid circles for now.
                x = int(ripple.cx + ripple.radius * math.cos(angle_rad))
                y = int(ripple.cy + ripple.radius * math.sin(angle_rad) * 0.6) # Y correction for char aspect

                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    char_grid[y][x] = (char_to_use, ripple.style)

        # Convert char_grid to styled output lines
        styled_output_lines = []
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                char, style = char_grid[r_idx][c_idx]
                escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                line_segments.append(f"[{style}]{escaped_char}[/{style}]")
            styled_output_lines.append("".join(line_segments))

        # Overlay title (centered)
        if self.title:
            title_y = self.display_height // 2
            title_x_start = (self.display_width - len(self.title)) // 2
            if 0 <= title_y < self.display_height:
                # Reconstruct the title line to overlay on top of ripples
                title_line_segments = []
                current_title_char_idx = 0
                for c_idx in range(self.display_width):
                    is_title_char_pos = title_x_start <= c_idx < title_x_start + len(self.title)
                    if is_title_char_pos:
                        char_to_draw = self.title[current_title_char_idx].replace('[', r'\[')
                        title_line_segments.append(f"[{self.title_style}]{char_to_draw}[/{self.title_style}]")
                        current_title_char_idx +=1
                    else: # Use the already determined char from char_grid (ripple or water)
                        char, style = char_grid[title_y][c_idx]
                        escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                        title_line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                styled_output_lines[title_y] = "".join(title_line_segments)

        return "\n".join(styled_output_lines)


class PixelZoomEffect(BaseEffect):
    """Starts with a pixelated (blocky) version of an ASCII art and resolves to clear."""

    def __init__(
        self,
        parent_widget: Any,
        target_content: str, # The clear, final ASCII art
        duration: float = 2.5, # Total duration of the effect
        max_pixel_size: int = 8, # Max block size for pixelation (e.g., 8x8 chars become one block)
        effect_type: str = "resolve", # "resolve" (pixelated to clear) or "pixelate" (clear to pixelated)
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.target_lines = target_content.splitlines()
        self.duration = duration
        self.max_pixel_size = max(1, max_pixel_size) # Must be at least 1
        self.effect_type = effect_type

        # Normalize target content dimensions
        self.content_height = len(self.target_lines)
        self.content_width = max(len(line) for line in self.target_lines) if self.target_lines else 0

        self.padded_target_lines = []
        if self.content_height > 0 and self.content_width > 0:
            for i in range(self.content_height):
                line = self.target_lines[i] if i < len(self.target_lines) else ""
                self.padded_target_lines.append(line + ' ' * (self.content_width - len(line)))
        else: # Handle empty target_content
            self.content_height = 1
            self.content_width = 1
            self.padded_target_lines = [" "]


    def _get_block_char(self, r_start: int, c_start: int, pixel_size: int) -> str:
        """Determines the representative character for a block."""
        if not self.padded_target_lines: return " "

        char_counts = {}
        num_chars_in_block = 0
        for r_offset in range(pixel_size):
            for c_offset in range(pixel_size):
                r, c = r_start + r_offset, c_start + c_offset
                if 0 <= r < self.content_height and 0 <= c < self.content_width:
                    char = self.padded_target_lines[r][c]
                    if char != ' ': # Ignore spaces for dominant char, or include if you want space to dominate
                        char_counts[char] = char_counts.get(char, 0) + 1
                        num_chars_in_block +=1

        if not char_counts: # Block is all spaces or out of bounds
            # Check the top-left char of the block in target art for a hint
            if 0 <= r_start < self.content_height and 0 <= c_start < self.content_width:
                 return self.padded_target_lines[r_start][c_start] # Could be a space
            return " "

        # Return the most frequent character in the block
        dominant_char = max(char_counts, key=char_counts.get)
        return dominant_char

    def update(self) -> Optional[str]:
        if not self.padded_target_lines : return " "

        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration)

        current_pixel_size = 1
        if self.effect_type == "resolve":
            # Pixel size decreases from max_pixel_size to 1
            # Using (1-progress) for size factor, so at progress=0, factor=1 (max size)
            # and at progress=1, factor=0 (min size = 1)
            size_factor = 1.0 - progress
            current_pixel_size = 1 + int(size_factor * (self.max_pixel_size - 1))
        elif self.effect_type == "pixelate":
            # Pixel size increases from 1 to max_pixel_size
            size_factor = progress
            current_pixel_size = 1 + int(size_factor * (self.max_pixel_size - 1))

        current_pixel_size = max(1, current_pixel_size) # Ensure it's at least 1

        if current_pixel_size == 1 and self.effect_type == "resolve":
            return "\n".join(self.padded_target_lines).replace('[',r'\[')
        if current_pixel_size == self.max_pixel_size and self.effect_type == "pixelate" and progress >=1.0:
            # Final pixelated state, render it once more and then could be static
             pass # Let it render below

        output_art_chars = [[' ' for _ in range(self.content_width)] for _ in range(self.content_height)]

        for r_block_start in range(0, self.content_height, current_pixel_size):
            for c_block_start in range(0, self.content_width, current_pixel_size):
                block_char = self._get_block_char(r_block_start, c_block_start, current_pixel_size)
                for r_offset in range(current_pixel_size):
                    for c_offset in range(current_pixel_size):
                        r, c = r_block_start + r_offset, c_block_start + c_offset
                        if 0 <= r < self.content_height and 0 <= c < self.content_width:
                            output_art_chars[r][c] = block_char

        # The card's base style will apply. No specific styling here unless needed.
        return "\n".join("".join(row) for row in output_art_chars).replace('[',r'\[')


class TextExplosionEffect(BaseEffect):
    """Animates characters of a text to explode outwards or implode inwards."""

    @dataclass
    class AnimatedChar:
        char: str
        origin_x: float # Final x in assembled text
        origin_y: float # Final y in assembled text
        current_x: float
        current_y: float
        vx: float # velocity for explosion (if random start)
        vy: float # velocity for explosion
        style: str = "bold white"

    def __init__(
        self,
        parent_widget: Any,
        text: str = "EXPLODE!",
        effect_type: str = "explode", # "explode" or "implode"
        duration: float = 1.5,
        width: int = 80, # Display area
        height: int = 24,
        char_style: str = "bold yellow",
        particle_spread: float = 30.0, # How far particles spread for explosion/implosion start
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.target_text = text
        self.effect_type = effect_type
        self.duration = duration
        self.display_width = width
        self.display_height = height
        self.char_style = char_style # Can be a list of styles too
        self.particle_spread = particle_spread

        self.chars: List[TextExplosionEffect.AnimatedChar] = []
        self._prepare_chars()

    def _prepare_chars(self):
        # Simple centered text for now
        text_width = len(self.target_text)
        start_x = (self.display_width - text_width) / 2.0
        origin_y = self.display_height / 2.0

        for i, char_val in enumerate(self.target_text):
            ox = start_x + i
            oy = origin_y

            if self.effect_type == "explode":
                # Start at origin, will move outwards
                cx, cy = ox, oy
                # Velocity is random, pointing outwards from text's rough center
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(self.particle_spread*0.5, self.particle_spread*1.5) / self.duration
                vx, vy = math.cos(angle) * speed, math.sin(angle) * speed * 0.5 # Y velocity halved for aspect
            else: # implode
                # Start at random spread-out position, move towards origin
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(self.particle_spread * 0.5, self.particle_spread)
                cx = ox + math.cos(angle) * dist
                cy = oy + math.sin(angle) * dist * 0.5 # Y spread halved
                vx, vy = 0,0 # Not used for velocity in implode, calculated by interpolation

            self.chars.append(TextExplosionEffect.AnimatedChar(
                char=char_val, origin_x=ox, origin_y=oy, current_x=cx, current_y=cy, vx=vx, vy=vy, style=self.char_style
            ))

    def update(self) -> Optional[str]:
        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration) # Normalized time 0 to 1

        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        # Store (char, style) to handle multiple chars on same spot (last one wins, or brightest)
        styled_grid: Dict[Tuple[int,int], Tuple[str,str]] = {}


        for achar in self.chars:
            if self.effect_type == "explode":
                # Move based on initial velocity, progress acting as time factor
                # This makes them fly out at constant speed after "explosion"
                # A better explosion might have initial burst then slowdown.
                # For now, constant velocity after start.
                if progress > 0.05 : # Small delay before explosion starts or to ensure movement
                    achar.current_x += achar.vx * progress * 20 # Scale velocity effect by progress and arbitrary factor
                    achar.current_y += achar.vy * progress * 20
            else: # implode
                # Interpolate from start (current_x, current_y at progress=0) to origin_x, origin_y
                # Initial current_x/y were set in _prepare_chars
                # We need to store the very initial random positions
                if not hasattr(achar, 'initial_x'): # Store initial random pos if not already
                    achar.initial_x = achar.current_x
                    achar.initial_y = achar.current_y

                # Interpolate based on progress. At progress=0, use initial. At progress=1, use origin.
                achar.current_x = achar.initial_x + (achar.origin_x - achar.initial_x) * progress
                achar.current_y = achar.initial_y + (achar.origin_y - achar.initial_y) * progress

            ix, iy = int(achar.current_x), int(achar.current_y)
            if 0 <= ix < self.display_width and 0 <= iy < self.display_height:
                # Simplistic: last char to land on a cell wins
                styled_grid[(ix,iy)] = (achar.char, achar.style)

        # Render styled_grid to output lines
        output_lines = []
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                if (c_idx, r_idx) in styled_grid:
                    char, style = styled_grid[(c_idx, r_idx)]
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(' ')
            output_lines.append("".join(line_segments))

        return "\n".join(output_lines)


class OldFilmEffect(BaseEffect):
    """Simulates an old film projector effect with shaky frames and film grain."""

    def __init__(
        self,
        parent_widget: Any,
        frames_content: List[str], # List of ASCII art strings, each a frame
        frame_duration: float = 0.5, # How long each frame stays before switching
        shake_intensity: int = 1, # Max character offset for shake (0 for no shake)
        grain_density: float = 0.05, # Chance for a character to be a grain speck
        grain_chars: str = ".:'",
        base_style: str = "sepia", # e.g., "sepia", "grayscale", or just "white on black"
        # Projector beam not implemented in this version for simplicity
        width: int = 80,
        height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.frames = [frame.splitlines() for frame in frames_content]
        if not self.frames: # Ensure there's at least one frame
            self.frames = [["Error: No frames provided".center(width)]]

        # Normalize all frames to consistent dimensions
        self.frame_height = max(len(f) for f in self.frames)
        self.frame_width = max(max(len(line) for line in f) if f else 0 for f in self.frames)

        padded_frames = []
        for frame_idx, frame_data in enumerate(self.frames):
            current_padded_frame = []
            for i in range(self.frame_height):
                line = frame_data[i] if i < len(frame_data) else ""
                current_padded_frame.append(line + ' ' * (self.frame_width - len(line)))
            padded_frames.append(current_padded_frame)
        self.frames = padded_frames

        self.frame_duration = frame_duration
        self.shake_intensity = shake_intensity
        self.grain_density = grain_density
        self.grain_chars = grain_chars
        self.base_style = base_style # This style will be applied to the frame content
        self.display_width = width
        self.display_height = height

        self.current_frame_index = 0
        self.time_on_current_frame = 0.0
        self.time_at_last_frame_render = time.time()

    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame_render
        self.time_at_last_frame_render = current_time
        self.time_on_current_frame += delta_time

        if self.time_on_current_frame >= self.frame_duration:
            self.current_frame_index = (self.current_frame_index + 1) % len(self.frames)
            self.time_on_current_frame = 0.0

        current_frame_data = self.frames[self.current_frame_index]

        # Apply shake
        dx, dy = 0, 0
        if self.shake_intensity > 0:
            dx = random.randint(-self.shake_intensity, self.shake_intensity)
            dy = random.randint(-self.shake_intensity, self.shake_intensity)

        # Prepare display grid (chars only first)
        # Output grid matches display_width, display_height
        # Frame content is centered within this.

        output_grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]

        frame_start_row = (self.display_height - self.frame_height) // 2 + dy
        frame_start_col = (self.display_width - self.frame_width) // 2 + dx

        for r_frame in range(self.frame_height):
            for c_frame in range(self.frame_width):
                r_disp, c_disp = frame_start_row + r_frame, frame_start_col + c_frame
                if 0 <= r_disp < self.display_height and 0 <= c_disp < self.display_width:
                    char_to_draw = current_frame_data[r_frame][c_frame]

                    # Apply film grain
                    if random.random() < self.grain_density:
                        char_to_draw = random.choice(self.grain_chars)

                    output_grid[r_disp][c_disp] = char_to_draw

        # Convert to styled lines
        styled_output_lines = []
        for r_idx in range(self.display_height):
            line_str = "".join(output_grid[r_idx]).replace('[',r'\[')
            # Apply base style to the whole line (simpler than per-char if base_style is uniform)
            styled_output_lines.append(f"[{self.base_style}]{line_str}[/{self.base_style}]")

        return "\n".join(styled_output_lines)


class GameOfLifeEffect(BaseEffect):
    """Simulates Conway's Game of Life."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "Evolving Systems...",
        width: int = 40, # Grid width for GoL (actual output width might be larger for title)
        height: int = 20, # Grid height for GoL
        cell_alive_char: str = "█",
        cell_dead_char: str = " ",
        alive_style: str = "bold green",
        dead_style: str = "black", # Effectively background
        initial_pattern: str = "random", # "random", or specific pattern names like "glider"
        update_interval: float = 0.2, # How often GoL updates, distinct from animation_speed
        title_style: str = "bold white",
        display_width: int = 80, # Total width for splash screen display
        display_height: int = 24, # Total height for splash screen display
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.grid_width = width
        self.grid_height = height
        self.cell_alive_char = cell_alive_char[0]
        self.cell_dead_char = cell_dead_char[0]
        self.alive_style = alive_style
        self.dead_style = dead_style # Usually background color
        self.initial_pattern = initial_pattern
        self.title_style = title_style
        self.display_width = display_width
        self.display_height = display_height

        self.grid = [[(random.choice([0,1]) if initial_pattern == "random" else 0) for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        if self.initial_pattern == "glider":
            # A common GoL pattern
            if self.grid_width >= 3 and self.grid_height >= 3:
                self.grid[0][1] = 1
                self.grid[1][2] = 1
                self.grid[2][0] = 1
                self.grid[2][1] = 1
                self.grid[2][2] = 1
        # Can add more predefined patterns here

        self._last_gol_update_time = time.time()
        self.gol_update_interval = update_interval


    def _count_neighbors(self, r: int, c: int) -> int:
        count = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                nr, nc = r + i, c + j
                # Toroidal grid (wraps around)
                nr = nr % self.grid_height
                nc = nc % self.grid_width
                count += self.grid[nr][nc]
        return count

    def _update_grid(self):
        new_grid = [[0 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                neighbors = self._count_neighbors(r, c)
                if self.grid[r][c] == 1: # Alive
                    if neighbors < 2 or neighbors > 3:
                        new_grid[r][c] = 0 # Dies
                    else:
                        new_grid[r][c] = 1 # Lives
                else: # Dead
                    if neighbors == 3:
                        new_grid[r][c] = 1 # Becomes alive
        self.grid = new_grid

    def update(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self._last_gol_update_time >= self.gol_update_interval:
            self._update_grid()
            self._last_gol_update_time = current_time

        # Render the grid and title to the full display_width/height
        output_display = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styled_output_lines = [""] * self.display_height

        # Center the GoL grid within the display area
        grid_start_r = (self.display_height - self.grid_height) // 2
        grid_start_c = (self.display_width - self.grid_width) // 2

        for r_grid in range(self.grid_height):
            for c_grid in range(self.grid_width):
                r_disp, c_disp = grid_start_r + r_grid, grid_start_c + c_grid
                if 0 <= r_disp < self.display_height and 0 <= c_disp < self.display_width:
                    is_alive = self.grid[r_grid][c_grid] == 1
                    char_to_draw = self.cell_alive_char if is_alive else self.cell_dead_char
                    style_to_use = self.alive_style if is_alive else self.dead_style
                    # Store as (char, style) tuple for later Rich formatting
                    output_display[r_disp][c_disp] = (char_to_draw, style_to_use)

        # Convert output_display (which has tuples or spaces) to styled lines
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                cell_content = output_display[r_idx][c_idx]
                if isinstance(cell_content, tuple):
                    char, style = cell_content
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else: # It's a space from initialization
                    line_segments.append(f"[{self.dead_style}] [/{self.dead_style}]") # Styled background space
            styled_output_lines[r_idx] = "".join(line_segments)


        # Overlay title (centered, typically above or below GoL grid)
        if self.title:
            title_y = grid_start_r - 2 if grid_start_r > 1 else self.display_height -1 # Place above or at bottom
            title_x_start = (self.display_width - len(self.title)) // 2

            if 0 <= title_y < self.display_height:
                # Construct title line, preserving background cells not covered by title
                title_line_segments = []
                title_char_idx = 0
                for c_idx in range(self.display_width):
                    is_title_char_pos = title_x_start <= c_idx < title_x_start + len(self.title)
                    if is_title_char_pos:
                        char = self.title[title_char_idx].replace('[',r'\[')
                        title_line_segments.append(f"[{self.title_style}]{char}[/{self.title_style}]")
                        title_char_idx += 1
                    else: # Part of the line not covered by title, use existing content
                        cell_content = output_display[title_y][c_idx]
                        if isinstance(cell_content, tuple):
                            char, style = cell_content
                            escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                            title_line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                        else:
                            title_line_segments.append(f"[{self.dead_style}] [/{self.dead_style}]")
                styled_output_lines[title_y] = "".join(title_line_segments)

        return "\n".join(styled_output_lines)


class ScrollingCreditsEffect(BaseEffect):
    """Simulates scrolling credits, like at the end of a movie."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        credits_list: List[Dict[str, str]] = None, # Each dict: {"role": "Concept", "name": "The Universe"} or just {"line": "Some text"}
        scroll_speed: float = 1.0, # Lines per second (can be fractional)
        line_spacing: int = 1, # Number of blank lines between credit entries
        width: int = 80,
        height: int = 24,
        title_style: str = "bold yellow",
        role_style: str = "bold white",
        name_style: str = "white",
        line_style: str = "white", # For single line credits
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.overall_title = title # Title for the splash screen itself, displayed statically
        self.credits_list = credits_list if credits_list else [{"line": "Loading..."}]
        self.scroll_speed = scroll_speed # This will be used to calculate fractional line shifts
        self.line_spacing = line_spacing
        self.display_width = width
        self.display_height = height
        self.title_style = title_style
        self.role_style = role_style
        self.name_style = name_style
        self.line_style = line_style

        self.formatted_credit_lines: List[str] = []
        self._format_credits()

        self.current_scroll_offset = float(self.display_height) # Start with credits off-screen at the bottom
        self.time_at_last_frame = time.time()


    def _format_credits(self):
        """Pre-formats credit entries into Rich-styled strings."""
        self.formatted_credit_lines.append(f"[{self.title_style}]{self.overall_title.center(self.display_width)}[/{self.title_style}]")
        self.formatted_credit_lines.append("") # Blank line after title

        for item in self.credits_list:
            if "line" in item:
                # Single line entry
                text = item["line"].replace('[', r'\[')
                self.formatted_credit_lines.append(f"[{self.line_style}]{text.center(self.display_width)}[/{self.line_style}]")
            elif "role" in item and "name" in item:
                # Role: Name format
                role_text = item["role"].replace('[', r'\[')
                name_text = item["name"].replace('[', r'\[')
                # Simple centered alignment for now
                # More complex alignment (role left, name right) is harder with fixed width and Rich.
                # For now, centered role, then centered name on next line or combined.
                # Let's do: Role (centered), Name (centered below it)
                self.formatted_credit_lines.append(f"[{self.role_style}]{role_text.center(self.display_width)}[/{self.role_style}]")
                self.formatted_credit_lines.append(f"[{self.name_style}]{name_text.center(self.display_width)}[/{self.name_style}]")

            for _ in range(self.line_spacing):
                self.formatted_credit_lines.append("") # Add blank lines for spacing

        # Add some padding at the end so last credit scrolls fully off
        for _ in range(self.display_height // 2):
            self.formatted_credit_lines.append("")


    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.time_at_last_frame
        self.time_at_last_frame = current_time

        self.current_scroll_offset -= delta_time * self.scroll_speed

        # Determine which lines are visible
        output_lines = []
        start_line_index = int(self.current_scroll_offset)

        for i in range(self.display_height):
            current_line_to_fetch = start_line_index + i
            if 0 <= current_line_to_fetch < len(self.formatted_credit_lines):
                output_lines.append(self.formatted_credit_lines[current_line_to_fetch])
            else:
                output_lines.append(' ' * self.display_width) # Blank line

        # Reset scroll if all credits have passed
        # Total height of credits content: len(self.formatted_credit_lines)
        # Resets when the top of the credits (index 0) has scrolled past the top of the screen (offset becomes negative enough)
        if self.current_scroll_offset < -len(self.formatted_credit_lines):
             self.current_scroll_offset = float(self.display_height) # Reset to start from bottom again

        return "\n".join(output_lines)


class PulseEffect(BaseEffect):
    """Creates a pulsing effect for text content by varying brightness."""

    def __init__(
        self,
        parent_widget: Any,
        content: str,
        pulse_speed: float = 0.1,
        min_brightness: int = 100,
        max_brightness: int = 255,
        color: Tuple[int, int, int] = (255, 255, 255),  # Default white
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.pulse_speed = pulse_speed  # Not directly used by time, but for step in cycle
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.base_color = color
        self._pulse_direction = 1
        self._current_brightness_step = 0
        self.num_steps = 20  # Number of steps from min to max brightness

    def update(self) -> Optional[str]:
        """Update the pulsing effect."""
        # Calculate brightness based on a sine wave for smooth pulsing
        # The frequency of the sine wave determines the pulse speed
        # time.time() - self.start_time gives elapsed time
        elapsed_time = time.time() - self.start_time

        # Create a sine wave that oscillates between 0 and 1
        # pulse_speed here could adjust the frequency of the sine wave
        # A higher pulse_speed value would make it pulse faster.
        # Let's define pulse_speed as cycles per second.
        # So, angle = 2 * pi * elapsed_time * pulse_speed
        # sin_value ranges from -1 to 1. We map it to 0 to 1.
        sin_value = (math.sin(2 * math.pi * elapsed_time * self.pulse_speed) + 1) / 2

        # Map the 0-1 sin_value to the brightness range
        brightness = self.min_brightness + (self.max_brightness - self.min_brightness) * sin_value
        brightness = int(max(0, min(255, brightness))) # Clamp

        # Apply brightness to the base color
        # Scale base color components by brightness/255
        r = int(self.base_color[0] * brightness / 255)
        g = int(self.base_color[1] * brightness / 255)
        b = int(self.base_color[2] * brightness / 255)

        pulsing_style = f"rgb({r},{g},{b})"

        # Apply style to content, escaping Rich markup
        output_lines = []
        for line in self.content.split('\n'):
            escaped_line = line.replace('[', r'\[').replace(']', r'\]')
            output_lines.append(f"[{pulsing_style}]{escaped_line}[/{pulsing_style}]")

        return '\n'.join(output_lines)


class BlinkEffect(BaseEffect):
    """Makes specified parts of the text blink."""

    def __init__(
        self,
        parent_widget: Any,
        content: str,
        blink_speed: float = 0.5,  # Time for one state (on or off)
        blink_targets: Optional[List[str]] = None, # List of exact strings to blink
        blink_style_on: str = "default", # Style when text is visible
        blink_style_off: str = "dim",  # Style when text is "off" (e.g., dimmed or hidden via color)
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.original_content = content
        self.blink_speed = blink_speed
        self.blink_targets = blink_targets if blink_targets else []
        self.blink_style_on = blink_style_on # Not actively used if "default" means use card's base style
        self.blink_style_off = blink_style_off
        self._is_on = True
        self._last_blink_time = time.time()

    def update(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self._last_blink_time >= self.blink_speed:
            self._is_on = not self._is_on
            self._last_blink_time = current_time

        # This is a simplified blink effect. For complex Rich text, direct string
        # manipulation of styled text is tricky. This version applies a style to target
        # strings when they should be "off" or replaces them.

        # Start with the original content, pre-escaped for Rich tags.
        # This assumes the card's main style will handle the "on" state appearance.
        # The effect focuses on altering the "off" state or replacing text.

        output_text = Text.from_markup(self.original_content.replace('[', r'\['))

        if not self._is_on:
            for target_text in self.blink_targets:
                # Find all occurrences of target_text and apply blink_style_off or hide
                start_index = 0
                while True:
                    try:
                        # Search in the plain string version of the Text object
                        found_pos = output_text.plain.find(target_text, start_index)
                        if found_pos == -1:
                            break

                        if self.blink_style_off == "hidden":
                            # Replace with spaces
                            output_text.plain = output_text.plain[:found_pos] + ' ' * len(target_text) + output_text.plain[found_pos+len(target_text):]
                            # This modification of .plain is a bit of a hack.
                            # A more robust way would be to reconstruct the Text object or use Text.replace.
                            # For now, let's rebuild the text object to ensure spans are cleared.
                            current_plain = output_text.plain
                            output_text = Text(current_plain) # Re-create to clear old spans over modified region
                        else:
                            # Apply style
                            output_text.stylize(self.blink_style_off, start=found_pos, end=found_pos + len(target_text))

                        start_index = found_pos + len(target_text)
                    except ValueError: # Should not happen with plain.find
                        break
        # If _is_on, the text remains as is, relying on the Static widget's base style.
        # If blink_style_on was not "default", one would apply it here to the targets.

        return output_text.markup # Return the Rich markup string


class CodeScrollEffect(BaseEffect):
    """Shows scrolling lines of pseudo-code with a title overlay."""

    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Initializing Systems...",
        width: int = 80, # Target width
        height: int = 24, # Target height
        scroll_speed: float = 0.1, # Affects how often lines shift
        num_code_lines: int = 15, # Number of visible code lines
        code_line_style: str = "dim cyan",
        title_style: str = "bold white",
        subtitle_style: str = "white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        self.scroll_speed = scroll_speed # Interpreted as interval for scrolling
        self.num_code_lines = min(num_code_lines, height -4) # Ensure space for title/subtitle
        self.code_line_style = code_line_style
        self.title_style = title_style
        self.subtitle_style = subtitle_style

        self.code_lines: List[str] = []
        self._last_scroll_time = time.time()
        self._generate_initial_code_lines()

    def _generate_random_code_line(self) -> str:
        """Generates a random line of pseudo-code."""
        line_len = random.randint(self.width // 2, self.width - 10)
        chars = "abcdef0123456789[];():=" + " " * 20 # More spaces
        line = "".join(random.choice(chars) for _ in range(line_len))
        # Add some indents
        indent = " " * random.randint(0, 8)
        return (indent + line)[:self.width]


    def _generate_initial_code_lines(self):
        for _ in range(self.num_code_lines):
            self.code_lines.append(self._generate_random_code_line())

    def update(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self._last_scroll_time >= self.scroll_speed:
            self.code_lines.pop(0)  # Remove oldest line
            self.code_lines.append(self._generate_random_code_line())  # Add new line
            self._last_scroll_time = current_time

        output_lines = []

        # Determine positions for title and subtitle
        code_block_start_row = (self.height - self.num_code_lines) // 2
        code_block_end_row = code_block_start_row + self.num_code_lines

        # Title position: centered, a few lines above the code block or screen center
        # Ensure it's within bounds and leaves space for subtitle if code block is small or high
        title_row_ideal = self.height // 2 - 3
        title_row = max(0, min(title_row_ideal, code_block_start_row - 2 if self.num_code_lines > 0 else title_row_ideal))

        # Subtitle position: centered, below title
        subtitle_row_ideal = title_row + 2
        subtitle_row = max(title_row + 1, min(subtitle_row_ideal, self.height -1))
        if subtitle_row >= code_block_start_row and self.num_code_lines > 0 : # Adjust if subtitle overlaps code block
             subtitle_row = min(self.height -1, code_block_start_row -1)
             if subtitle_row <= title_row: # if code block is too high, push title up
                 title_row = max(0, subtitle_row - 2)


        for r_idx in range(self.height):
            current_line_content = ""
            if r_idx == title_row:
                padding = (self.width - len(self.title)) // 2
                current_line_content = f"{' ' * padding}{self.title}{' ' * (self.width - len(self.title) - padding)}"
                escaped_content = current_line_content.replace('[', ESCAPED_OPEN_BRACKET)
                current_line_content = f"[{self.title_style}]{escaped_content}[/{self.title_style}]"
            elif r_idx == subtitle_row:
                padding = (self.width - len(self.subtitle)) // 2
                current_line_content = f"{' ' * padding}{self.subtitle}{' ' * (self.width - len(self.subtitle) - padding)}"
                escaped_content = current_line_content.replace('[', ESCAPED_OPEN_BRACKET)
                current_line_content = f"[{self.subtitle_style}]{escaped_content}[/{self.subtitle_style}]"
            elif code_block_start_row <= r_idx < code_block_end_row:
                code_line_index = r_idx - code_block_start_row
                code_text = self.code_lines[code_line_index]
                # Ensure code_text is padded to full width if needed, or handled by terminal
                escaped_code = code_text.replace('[', ESCAPED_OPEN_BRACKET)
                current_line_content = f"[{self.code_line_style}]{escaped_code}{' ' * (self.width - len(code_text))}[/{self.code_line_style}]"
            else:
                current_line_content = ' ' * self.width

            output_lines.append(current_line_content)

        return '\n'.join(output_lines)


class WaveRippleEffect(BaseEffect):
    """Creates expanding ripple patterns from the center outward, like dropping a stone in water."""
    
    @dataclass
    class Wave:
        radius: float
        max_radius: float
        amplitude: float  # Height of the wave
        speed: float
        color_index: int
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Rippling through data...",
        width: int = 80,
        height: int = 24,
        wave_speed: float = 10.0,  # Pixels per second
        wave_spawn_rate: float = 1.0,  # New waves per second
        max_waves: int = 5,
        wave_chars: str = "·∙○◯◉●",
        wave_colors: List[str] = ["dim cyan", "cyan", "bright_cyan", "white", "cyan", "dim cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.wave_speed = wave_speed
        self.wave_spawn_rate = wave_spawn_rate
        self.max_waves = max_waves
        self.wave_chars = wave_chars
        self.wave_colors = wave_colors
        self.title_style = title_style
        
        self.waves: List[WaveRippleEffect.Wave] = []
        self.time_since_last_wave = 0.0
        self.last_update_time = time.time()
        
        # Center point for ripples
        self.center_x = width / 2
        self.center_y = height / 2
        
    def _spawn_wave(self):
        """Spawn a new wave from the center."""
        if len(self.waves) < self.max_waves:
            max_radius = math.sqrt((self.display_width/2)**2 + (self.display_height/2)**2)
            self.waves.append(WaveRippleEffect.Wave(
                radius=0.0,
                max_radius=max_radius,
                amplitude=1.0,
                speed=self.wave_speed,
                color_index=0
            ))
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Spawn new waves
        self.time_since_last_wave += delta_time
        if self.time_since_last_wave >= 1.0 / self.wave_spawn_rate:
            self._spawn_wave()
            self.time_since_last_wave = 0.0
        
        # Update existing waves
        active_waves = []
        for wave in self.waves:
            wave.radius += wave.speed * delta_time
            if wave.radius < wave.max_radius:
                # Update amplitude (fade out as it expands)
                wave.amplitude = max(0, 1.0 - (wave.radius / wave.max_radius))
                # Update color index based on radius
                wave.color_index = int((wave.radius / wave.max_radius) * (len(self.wave_colors) - 1))
                active_waves.append(wave)
        self.waves = active_waves
        
        # Render waves
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw waves (older/larger ones first)
        for wave in sorted(self.waves, key=lambda w: w.radius, reverse=True):
            # Draw circle at current radius
            num_points = int(2 * math.pi * wave.radius) + 1
            if num_points < 4:
                num_points = 4
                
            for i in range(num_points):
                angle = (2 * math.pi * i) / num_points
                x = int(self.center_x + wave.radius * math.cos(angle))
                y = int(self.center_y + wave.radius * math.sin(angle) * 0.5)  # Aspect ratio correction
                
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    # Choose character based on wave amplitude
                    char_index = int((1.0 - wave.amplitude) * (len(self.wave_chars) - 1))
                    char_index = max(0, min(char_index, len(self.wave_chars) - 1))
                    
                    grid[y][x] = self.wave_chars[char_index]
                    styles[y][x] = self.wave_colors[wave.color_index]
        
        # Overlay title and subtitle
        if self.title:
            title_y = int(self.center_y) - 2
            title_x = int((self.display_width - len(self.title)) / 2)
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        if self.subtitle:
            subtitle_y = int(self.center_y) + 1
            subtitle_x = int((self.display_width - len(self.subtitle)) / 2)
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class DNAHelixEffect(BaseEffect):
    """Animated double helix structure rotating in 3D ASCII art."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Decoding sequences...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.5,  # Rotations per second
        helix_width: int = 30,
        helix_height: int = 16,
        strand_chars: str = "◉●○",
        base_pair_chars: str = "═──",
        strand_colors: List[str] = ["red", "blue"],
        base_colors: List[str] = ["yellow", "green", "cyan", "magenta"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.helix_width = helix_width
        self.helix_height = helix_height
        self.strand_chars = strand_chars
        self.base_pair_chars = base_pair_chars
        self.strand_colors = strand_colors
        self.base_colors = base_colors
        self.title_style = title_style
        
        self.rotation = 0.0
        self.last_update_time = time.time()
        
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update rotation
        self.rotation += self.rotation_speed * delta_time * 2 * math.pi
        
        # Create display grid
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Calculate helix position
        helix_start_x = (self.display_width - self.helix_width) // 2
        helix_start_y = (self.display_height - self.helix_height) // 2
        
        # Draw DNA helix
        for y in range(self.helix_height):
            # Calculate the phase for this height
            phase1 = self.rotation + (y * 0.4)  # First strand
            phase2 = phase1 + math.pi  # Second strand (180 degrees offset)
            
            # Calculate x positions for both strands
            x1 = int(helix_start_x + self.helix_width/2 + (self.helix_width/2 - 2) * math.sin(phase1))
            x2 = int(helix_start_x + self.helix_width/2 + (self.helix_width/2 - 2) * math.sin(phase2))
            
            # Determine which strand is in front based on sin values
            z1 = math.cos(phase1)
            z2 = math.cos(phase2)
            
            display_y = helix_start_y + y
            
            if 0 <= display_y < self.display_height:
                # Draw base pair connection
                if abs(z1 - z2) < 1.8:  # Strands are close enough in z-plane
                    min_x = min(x1, x2)
                    max_x = max(x1, x2)
                    
                    # Draw connecting base pairs
                    for x in range(min_x + 1, max_x):
                        if 0 <= x < self.display_width:
                            # Choose base pair character based on position
                            char_index = ((x - min_x - 1) * len(self.base_pair_chars)) // (max_x - min_x - 1)
                            char_index = max(0, min(char_index, len(self.base_pair_chars) - 1))
                            
                            grid[display_y][x] = self.base_pair_chars[char_index]
                            # Cycle through base colors
                            color_index = y % len(self.base_colors)
                            styles[display_y][x] = self.base_colors[color_index]
                
                # Draw strands (front strand drawn last to appear on top)
                strands = [(x1, z1, 0), (x2, z2, 1)]
                for x, z, strand_index in sorted(strands, key=lambda s: s[1]):
                    if 0 <= x < self.display_width:
                        # Choose character based on z-depth
                        char_index = int((z + 1) * (len(self.strand_chars) - 1) / 2)
                        char_index = max(0, min(char_index, len(self.strand_chars) - 1))
                        
                        grid[display_y][x] = self.strand_chars[char_index]
                        styles[display_y][x] = self.strand_colors[strand_index]
        
        # Overlay title
        if self.title:
            title_y = helix_start_y - 2
            if title_y < 1:
                title_y = helix_start_y + self.helix_height + 1
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Overlay subtitle
        if self.subtitle:
            subtitle_y = helix_start_y + self.helix_height + 2
            if subtitle_y >= self.display_height - 1:
                subtitle_y = helix_start_y - 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)



class FireworksEffect(BaseEffect):
    """Simulates fireworks explosions using ASCII characters."""
    
    @dataclass
    class Firework:
        x: float
        y: float
        vx: float  # Initial velocity x
        vy: float  # Initial velocity y
        age: float
        max_age: float
        exploded: bool
        color: str
        particles: List[Tuple[float, float, float, float]]  # x, y, vx, vy
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Celebrating innovation...",
        width: int = 80,
        height: int = 24,
        launch_rate: float = 0.5,  # Fireworks per second
        gravity: float = 15.0,
        explosion_chars: str = "*+x·°",
        trail_char: str = "|",
        colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan", "white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.launch_rate = launch_rate
        self.gravity = gravity
        self.explosion_chars = explosion_chars
        self.trail_char = trail_char
        self.colors = colors
        self.title_style = title_style
        
        self.fireworks: List[FireworksEffect.Firework] = []
        self.time_since_last_launch = 0.0
        self.last_update_time = time.time()
        self.revealed_chars = set()  # Track which title characters have been revealed
        
    def _launch_firework(self):
        """Launch a new firework from the bottom of the screen."""
        x = random.randint(10, self.display_width - 10)
        y = self.display_height - 1
        vx = random.uniform(-2, 2)
        vy = random.uniform(-20, -15)  # Negative for upward motion
        max_age = random.uniform(0.8, 1.5)
        color = random.choice(self.colors)
        
        self.fireworks.append(FireworksEffect.Firework(
            x=float(x), y=float(y), vx=vx, vy=vy, 
            age=0.0, max_age=max_age, exploded=False,
            color=color, particles=[]
        ))
    
    def _explode_firework(self, fw: 'FireworksEffect.Firework'):
        """Create explosion particles."""
        num_particles = random.randint(20, 40)
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(5, 15)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle) * 0.5  # Aspect ratio correction
            fw.particles.append((fw.x, fw.y, vx, vy))
        fw.exploded = True
        
        # Check if explosion reveals any title characters
        title_y = self.display_height // 2 - 2
        title_x = (self.display_width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            char_x = title_x + i
            char_y = title_y
            # If explosion is near this character, reveal it
            if abs(fw.x - char_x) < 10 and abs(fw.y - char_y) < 6:
                self.revealed_chars.add(i)
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Launch new fireworks
        self.time_since_last_launch += delta_time
        if self.time_since_last_launch >= 1.0 / self.launch_rate:
            self._launch_firework()
            self.time_since_last_launch = 0.0
        
        # Update existing fireworks
        active_fireworks = []
        for fw in self.fireworks:
            fw.age += delta_time
            
            if fw.age < fw.max_age * 2:  # Keep for a bit after explosion
                if not fw.exploded:
                    # Update position
                    fw.x += fw.vx * delta_time
                    fw.y += fw.vy * delta_time
                    fw.vy += self.gravity * delta_time  # Apply gravity
                    
                    # Check if it's time to explode
                    if fw.age >= fw.max_age or fw.vy > 0:
                        self._explode_firework(fw)
                
                # Update particles
                if fw.exploded:
                    new_particles = []
                    for px, py, pvx, pvy in fw.particles:
                        px += pvx * delta_time
                        py += pvy * delta_time
                        pvy += self.gravity * delta_time * 0.5  # Less gravity for particles
                        
                        if 0 <= px < self.display_width and 0 <= py < self.display_height:
                            new_particles.append((px, py, pvx, pvy))
                    fw.particles = new_particles
                
                if fw.exploded and len(fw.particles) > 0:
                    active_fireworks.append(fw)
                elif not fw.exploded:
                    active_fireworks.append(fw)
        
        self.fireworks = active_fireworks
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw fireworks
        for fw in self.fireworks:
            if not fw.exploded:
                # Draw trail
                x, y = int(fw.x), int(fw.y)
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    grid[y][x] = self.trail_char
                    styles[y][x] = fw.color
            else:
                # Draw explosion particles
                for px, py, _, _ in fw.particles:
                    x, y = int(px), int(py)
                    if 0 <= x < self.display_width and 0 <= y < self.display_height:
                        # Choose character based on particle age
                        char_index = int((fw.age - fw.max_age) / fw.max_age * len(self.explosion_chars))
                        char_index = max(0, min(char_index, len(self.explosion_chars) - 1))
                        
                        grid[y][x] = self.explosion_chars[char_index]
                        styles[y][x] = fw.color
        
        # Draw title (revealed characters only)
        if self.title:
            title_y = self.display_height // 2 - 2
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                if i in self.revealed_chars:
                    x = title_x + i
                    if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                        grid[title_y][x] = char
                        styles[title_y][x] = self.title_style
        
        # Always show subtitle
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class CircuitBoardEffect(BaseEffect):
    """Animated circuit board traces that light up and connect different components."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Connecting systems...",
        width: int = 80,
        height: int = 24,
        trace_speed: float = 20.0,  # Characters per second
        trace_chars: str = "─│┌┐└┘├┤┬┴┼",
        node_chars: str = "◆◇○●□■",
        active_color: str = "bright_green",
        inactive_color: str = "dim green",
        node_color: str = "yellow",
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.trace_speed = trace_speed
        self.trace_chars = trace_chars
        self.node_chars = node_chars
        self.active_color = active_color
        self.inactive_color = inactive_color
        self.node_color = node_color
        self.title_style = title_style
        
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
        self.active_cells = set()
        self.nodes = []
        self.traces = []
        self.current_trace_progress = 0.0
        self.last_update_time = time.time()
        
        self._generate_circuit()
        
    def _generate_circuit(self):
        """Generate a random circuit board layout."""
        # Place nodes
        num_nodes = random.randint(6, 10)
        for _ in range(num_nodes):
            x = random.randint(5, self.display_width - 5)
            y = random.randint(2, self.display_height - 3)
            node_char = random.choice(self.node_chars)
            self.nodes.append((x, y, node_char))
            self.grid[y][x] = node_char
        
        # Connect nodes with traces
        for i in range(len(self.nodes) - 1):
            start = self.nodes[i]
            end = self.nodes[i + 1]
            path = self._create_path(start[:2], end[:2])
            if path:
                self.traces.append(path)
    
    def _create_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a path between two points using Manhattan routing."""
        x1, y1 = start
        x2, y2 = end
        path = []
        
        # Simple L-shaped path
        if random.random() < 0.5:
            # Horizontal first
            for x in range(min(x1, x2), max(x1, x2) + 1):
                path.append((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                path.append((x2, y))
        else:
            # Vertical first
            for y in range(min(y1, y2), max(y1, y2) + 1):
                path.append((x1, y))
            for x in range(min(x1, x2), max(x1, x2) + 1):
                path.append((x, y2))
        
        return path
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update trace animation
        self.current_trace_progress += self.trace_speed * delta_time
        
        # Calculate which cells should be active
        total_cells = sum(len(trace) for trace in self.traces)
        active_count = int(self.current_trace_progress) % (total_cells + 20)  # Add pause
        
        self.active_cells.clear()
        cell_count = 0
        for trace in self.traces:
            for x, y in trace:
                if cell_count < active_count:
                    self.active_cells.add((x, y))
                cell_count += 1
        
        # Render
        display_grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw traces
        for trace_idx, trace in enumerate(self.traces):
            for i, (x, y) in enumerate(trace):
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    # Determine trace character based on connections
                    if i == 0 or i == len(trace) - 1:
                        continue  # Skip nodes
                    
                    prev_x, prev_y = trace[i-1] if i > 0 else (x, y)
                    next_x, next_y = trace[i+1] if i < len(trace)-1 else (x, y)
                    
                    # Choose appropriate character
                    if prev_x == next_x:  # Vertical
                        char = '│'
                    elif prev_y == next_y:  # Horizontal
                        char = '─'
                    elif (prev_x < x and next_y < y) or (prev_y < y and next_x < x):
                        char = '┌'
                    elif (prev_x > x and next_y < y) or (prev_y < y and next_x > x):
                        char = '┐'
                    elif (prev_x < x and next_y > y) or (prev_y > y and next_x < x):
                        char = '└'
                    else:
                        char = '┘'
                    
                    display_grid[y][x] = char
                    styles[y][x] = self.active_color if (x, y) in self.active_cells else self.inactive_color
        
        # Draw nodes
        for x, y, char in self.nodes:
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                display_grid[y][x] = char
                styles[y][x] = self.node_color
        
        # Draw title
        if self.title:
            title_y = 1
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    display_grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = self.display_height - 2
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    display_grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = display_grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class PixelDissolveEffect(BaseEffect):
    """The screen starts filled with random ASCII characters that gradually dissolve away."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Revealing clarity...",
        width: int = 80,
        height: int = 24,
        dissolve_rate: float = 0.02,  # Percentage per frame
        noise_chars: str = "█▓▒░╳╱╲┃━┏┓┗┛",
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.dissolve_rate = dissolve_rate
        self.noise_chars = noise_chars
        self.title_style = title_style
        
        # Initialize with all pixels as noise
        self.dissolved_pixels = set()
        self.total_pixels = width * height
        
    def update(self) -> Optional[str]:
        # Calculate how many pixels to dissolve this frame
        current_dissolved = len(self.dissolved_pixels)
        target_dissolved = min(self.total_pixels, 
                              current_dissolved + int(self.total_pixels * self.dissolve_rate))
        
        # Dissolve random pixels
        while len(self.dissolved_pixels) < target_dissolved:
            x = random.randint(0, self.display_width - 1)
            y = random.randint(0, self.display_height - 1)
            self.dissolved_pixels.add((x, y))
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Fill with noise or clear based on dissolved state
        for y in range(self.display_height):
            for x in range(self.display_width):
                if (x, y) not in self.dissolved_pixels:
                    grid[y][x] = random.choice(self.noise_chars)
                    styles[y][x] = random.choice(["dim white", "dim gray", "dim black"])
        
        # Always show title and subtitle (on top of noise)
        if self.title:
            title_y = self.display_height // 2 - 2
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class TetrisBlockEffect(BaseEffect):
    """Tetris-style blocks fall from the top and stack up to form the title text."""
    
    @dataclass
    class Block:
        x: int
        y: float
        char: str
        target_y: int
        color: str
        falling: bool = True
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Building blocks...",
        width: int = 80,
        height: int = 24,
        fall_speed: float = 8.0,  # Blocks per second
        block_chars: str = "█",
        colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.fall_speed = fall_speed
        self.block_chars = block_chars
        self.colors = colors
        self.title_style = title_style
        
        self.blocks: List[TetrisBlockEffect.Block] = []
        self.last_update_time = time.time()
        self.spawn_delay = 0.1
        self.time_since_spawn = 0.0
        self.title_positions = []
        
        self._calculate_title_positions()
        self.spawn_index = 0
        
    def _calculate_title_positions(self):
        """Calculate where each character of the title should be."""
        title_y = self.display_height // 2 - 2
        title_x = (self.display_width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            if char != ' ':
                self.title_positions.append((title_x + i, title_y, char))
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Spawn new blocks
        self.time_since_spawn += delta_time
        if self.time_since_spawn >= self.spawn_delay and self.spawn_index < len(self.title_positions):
            x, y, char = self.title_positions[self.spawn_index]
            color = random.choice(self.colors)
            self.blocks.append(TetrisBlockEffect.Block(
                x=x, y=0, char=char, target_y=y, color=color
            ))
            self.spawn_index += 1
            self.time_since_spawn = 0.0
        
        # Update falling blocks
        for block in self.blocks:
            if block.falling:
                block.y += self.fall_speed * delta_time
                if block.y >= block.target_y:
                    block.y = block.target_y
                    block.falling = False
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw blocks
        for block in self.blocks:
            y = int(block.y)
            if 0 <= block.x < self.display_width and 0 <= y < self.display_height:
                grid[y][block.x] = block.char
                styles[y][block.x] = block.color if block.falling else self.title_style
        
        # Always show subtitle
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class SpiralGalaxyEffect(BaseEffect):
    """Creates a rotating spiral galaxy pattern with ASCII stars and cosmic dust."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Exploring the cosmos...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.2,  # Radians per second
        num_arms: int = 3,
        star_chars: str = "·∙*✦✧★☆",
        star_colors: List[str] = ["white", "bright_white", "yellow", "cyan", "dim white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.num_arms = num_arms
        self.star_chars = star_chars
        self.star_colors = star_colors
        self.title_style = title_style
        
        self.rotation = 0.0
        self.last_update_time = time.time()
        self.stars = []
        
        self._generate_stars()
        
    def _generate_stars(self):
        """Generate stars in spiral pattern."""
        center_x = self.display_width / 2
        center_y = self.display_height / 2
        
        for i in range(200):  # Number of stars
            # Spiral parameters
            angle = random.uniform(0, 4 * math.pi)
            radius = random.uniform(0, min(center_x, center_y) - 5)
            
            # Add arm structure
            arm = random.randint(0, self.num_arms - 1)
            arm_angle = (2 * math.pi * arm) / self.num_arms
            
            # Logarithmic spiral
            spiral_angle = angle + arm_angle + (radius * 0.1)
            
            x = center_x + radius * math.cos(spiral_angle)
            y = center_y + radius * math.sin(spiral_angle) * 0.5  # Aspect ratio
            
            char = random.choice(self.star_chars)
            color = random.choice(self.star_colors)
            
            self.stars.append({
                'radius': radius,
                'angle': spiral_angle,
                'char': char,
                'color': color,
                'brightness': random.uniform(0.3, 1.0)
            })
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.rotation += self.rotation_speed * delta_time
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width / 2
        center_y = self.display_height / 2
        
        # Draw stars
        for star in self.stars:
            # Rotate star
            angle = star['angle'] + self.rotation
            x = int(center_x + star['radius'] * math.cos(angle))
            y = int(center_y + star['radius'] * math.sin(angle) * 0.5)
            
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                # Twinkle effect
                if random.random() < star['brightness']:
                    grid[y][x] = star['char']
                    styles[y][x] = star['color']
        
        # Draw title emerging from galactic center
        if self.title:
            title_y = int(center_y)
            title_x = int((self.display_width - len(self.title)) / 2)
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = int(center_y) + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class MorphingShapeEffect(BaseEffect):
    """Geometric ASCII shapes that morph and transform into one another."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Transforming reality...",
        width: int = 80,
        height: int = 24,
        morph_speed: float = 0.5,  # Morphs per second
        shape_chars: str = "○□△▽◇◆",
        shape_colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.morph_speed = morph_speed
        self.shape_chars = shape_chars
        self.shape_colors = shape_colors
        self.title_style = title_style
        
        self.morph_progress = 0.0
        self.current_shape = 0
        self.shapes = ['circle', 'square', 'triangle', 'diamond']
        self.last_update_time = time.time()
        
    def _draw_circle(self, center_x: int, center_y: int, radius: int, progress: float):
        """Draw a circle shape."""
        points = []
        num_points = int(50 * progress)
        for i in range(num_points):
            angle = (2 * math.pi * i) / 50
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle) * 0.5)
            points.append((x, y))
        return points
    
    def _draw_square(self, center_x: int, center_y: int, size: int, progress: float):
        """Draw a square shape."""
        points = []
        half_size = size // 2
        perimeter = size * 4
        points_to_draw = int(perimeter * progress)
        
        for i in range(points_to_draw):
            if i < size:  # Top edge
                points.append((center_x - half_size + i, center_y - half_size))
            elif i < size * 2:  # Right edge
                points.append((center_x + half_size, center_y - half_size + (i - size)))
            elif i < size * 3:  # Bottom edge
                points.append((center_x + half_size - (i - size * 2), center_y + half_size))
            else:  # Left edge
                points.append((center_x - half_size, center_y + half_size - (i - size * 3)))
        
        return points
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.morph_progress += self.morph_speed * delta_time
        
        if self.morph_progress >= 1.0:
            self.morph_progress = 0.0
            self.current_shape = (self.current_shape + 1) % len(self.shapes)
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        
        # Draw current shape
        shape_type = self.shapes[self.current_shape]
        if shape_type == 'circle':
            points = self._draw_circle(center_x, center_y, 10, self.morph_progress)
        elif shape_type == 'square':
            points = self._draw_square(center_x, center_y, 20, self.morph_progress)
        else:
            points = []
        
        # Draw points
        for x, y in points:
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                char_index = int(self.morph_progress * (len(self.shape_chars) - 1))
                grid[y][x] = self.shape_chars[char_index]
                color_index = self.current_shape % len(self.shape_colors)
                styles[y][x] = self.shape_colors[color_index]
        
        # Draw title (always visible)
        if self.title:
            title_y = center_y
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = center_y + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class ParticleSwarmEffect(BaseEffect):
    """A swarm of ASCII characters that move with flocking behavior."""
    
    @dataclass
    class Particle:
        x: float
        y: float
        vx: float
        vy: float
        char: str
        color: str
        target_index: Optional[int] = None
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Swarming intelligence...",
        width: int = 80,
        height: int = 24,
        num_particles: int = 50,
        swarm_speed: float = 10.0,
        cohesion: float = 0.01,
        separation: float = 0.1,
        alignment: float = 0.05,
        particle_chars: str = "·∙○◦",
        particle_colors: List[str] = ["cyan", "blue", "white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.num_particles = num_particles
        self.swarm_speed = swarm_speed
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.particle_chars = particle_chars
        self.particle_colors = particle_colors
        self.title_style = title_style
        
        self.particles: List[ParticleSwarmEffect.Particle] = []
        self.last_update_time = time.time()
        self.formation_mode = False
        self.formation_progress = 0.0
        
        self._init_particles()
        self._calculate_title_positions()
        
    def _init_particles(self):
        """Initialize particles with random positions and velocities."""
        for i in range(self.num_particles):
            self.particles.append(ParticleSwarmEffect.Particle(
                x=random.uniform(0, self.display_width),
                y=random.uniform(0, self.display_height),
                vx=random.uniform(-2, 2),
                vy=random.uniform(-2, 2),
                char=random.choice(self.particle_chars),
                color=random.choice(self.particle_colors)
            ))
    
    def _calculate_title_positions(self):
        """Calculate target positions for title formation."""
        self.title_positions = []
        title_y = self.display_height // 2
        title_x = (self.display_width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            if char != ' ':
                self.title_positions.append((title_x + i, title_y))
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Switch to formation mode after some time
        elapsed = current_time - self.start_time
        if elapsed > 3.0 and not self.formation_mode:
            self.formation_mode = True
            # Assign particles to title positions
            for i, particle in enumerate(self.particles[:len(self.title_positions)]):
                particle.target_index = i
        
        # Update particles
        for particle in self.particles:
            if self.formation_mode and particle.target_index is not None:
                # Move towards target position
                target_x, target_y = self.title_positions[particle.target_index]
                dx = target_x - particle.x
                dy = target_y - particle.y
                
                particle.vx = dx * 0.1
                particle.vy = dy * 0.1
            else:
                # Flocking behavior
                # Find nearby particles
                neighbors = []
                for other in self.particles:
                    if other != particle:
                        dist = math.sqrt((other.x - particle.x)**2 + (other.y - particle.y)**2)
                        if dist < 10:
                            neighbors.append(other)
                
                if neighbors:
                    # Cohesion
                    avg_x = sum(n.x for n in neighbors) / len(neighbors)
                    avg_y = sum(n.y for n in neighbors) / len(neighbors)
                    particle.vx += (avg_x - particle.x) * self.cohesion
                    particle.vy += (avg_y - particle.y) * self.cohesion
                    
                    # Separation
                    for neighbor in neighbors:
                        dist = math.sqrt((neighbor.x - particle.x)**2 + (neighbor.y - particle.y)**2)
                        if dist < 5 and dist > 0:
                            particle.vx -= (neighbor.x - particle.x) / dist * self.separation
                            particle.vy -= (neighbor.y - particle.y) / dist * self.separation
                    
                    # Alignment
                    avg_vx = sum(n.vx for n in neighbors) / len(neighbors)
                    avg_vy = sum(n.vy for n in neighbors) / len(neighbors)
                    particle.vx += (avg_vx - particle.vx) * self.alignment
            
            # Limit speed
            speed = math.sqrt(particle.vx**2 + particle.vy**2)
            if speed > self.swarm_speed:
                particle.vx = (particle.vx / speed) * self.swarm_speed
                particle.vy = (particle.vy / speed) * self.swarm_speed
            
            # Update position
            particle.x += particle.vx * delta_time
            particle.y += particle.vy * delta_time
            
            # Wrap around edges
            particle.x = particle.x % self.display_width
            particle.y = particle.y % self.display_height
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw particles
        for particle in self.particles:
            x, y = int(particle.x), int(particle.y)
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                if self.formation_mode and particle.target_index is not None and particle.target_index < len(self.title):
                    grid[y][x] = self.title[particle.target_index]
                    styles[y][x] = self.title_style
                else:
                    grid[y][x] = particle.char
                    styles[y][x] = particle.color
        
        # Always show subtitle
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class ASCIIKaleidoscopeEffect(BaseEffect):
    """Creates symmetrical, rotating kaleidoscope patterns using ASCII characters."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Infinite patterns...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.3,
        num_mirrors: int = 6,
        pattern_chars: str = "◆◇○●□■▲▼",
        pattern_colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan", "white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.num_mirrors = num_mirrors
        self.pattern_chars = pattern_chars
        self.pattern_colors = pattern_colors
        self.title_style = title_style
        
        self.rotation = 0.0
        self.last_update_time = time.time()
        self.pattern_elements = []
        
        self._generate_pattern()
        
    def _generate_pattern(self):
        """Generate random pattern elements in one segment."""
        segment_angle = (2 * math.pi) / self.num_mirrors
        
        for _ in range(10):  # Number of elements per segment
            angle = random.uniform(0, segment_angle)
            radius = random.uniform(3, 15)
            char = random.choice(self.pattern_chars)
            color = random.choice(self.pattern_colors)
            
            self.pattern_elements.append({
                'angle': angle,
                'radius': radius,
                'char': char,
                'color': color
            })
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.rotation += self.rotation_speed * delta_time
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        
        # Draw kaleidoscope pattern
        for element in self.pattern_elements:
            for mirror in range(self.num_mirrors):
                # Calculate mirrored angle
                mirror_angle = (2 * math.pi * mirror) / self.num_mirrors
                angle = element['angle'] + mirror_angle + self.rotation
                
                # Calculate position
                x = int(center_x + element['radius'] * math.cos(angle))
                y = int(center_y + element['radius'] * math.sin(angle) * 0.5)
                
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    grid[y][x] = element['char']
                    styles[y][x] = element['color']
                
                # Add reflection
                if mirror % 2 == 0:
                    angle_reflected = -element['angle'] + mirror_angle + self.rotation
                    x = int(center_x + element['radius'] * math.cos(angle_reflected))
                    y = int(center_y + element['radius'] * math.sin(angle_reflected) * 0.5)
                    
                    if 0 <= x < self.display_width and 0 <= y < self.display_height:
                        grid[y][x] = element['char']
                        styles[y][x] = element['color']
        
        # Draw title in center
        if self.title:
            title_y = center_y
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = center_y + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class MiningEffect(BaseEffect):
    """Creates a Dwarf Fortress-style mining/digging animation."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        width: int = 80,
        height: int = 24,
        dig_speed: float = 0.8,  # How fast the mining progresses
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content_lines = content.strip().split('\n')
        self.width = width
        self.height = height
        self.dig_speed = dig_speed
        
        # Mining characters progression: stone -> partially dug -> empty
        self.stone_chars = ['█', '▓', '▒', '░', ' ']
        self.stone_styles = ['rgb(100,60,20)', 'rgb(120,80,40)', 'rgb(140,100,60)', 'rgb(160,120,80)', 'white']
        
        # Initialize mining grid - start with all stone
        self.mining_grid = [[0 for _ in range(width)] for _ in range(height)]
        self.reveal_grid = [[False for _ in range(width)] for _ in range(height)]
        
        # Track mining progress
        self.mining_progress = 0.0
        self.total_cells = sum(len(line) for line in self.content_lines)
        
        # Pickaxe animation
        self.pickaxe_pos = [width // 2, height // 2]
        self.pickaxe_char = '⚒'
        self.pickaxe_style = 'bold yellow'
        
        # Sound effect characters
        self.sound_effects = ['*', '∆', '◊', '✦']
        self.sound_positions = []
        
    def update(self) -> Optional[str]:
        """Update mining animation."""
        elapsed = time.time() - self.start_time
        
        # Progress mining based on elapsed time
        target_progress = min(1.0, elapsed * self.dig_speed / 5.0)  # 5 seconds total
        
        # Mine out cells progressively
        if target_progress > self.mining_progress:
            self.mining_progress = target_progress
            self._advance_mining()
        
        # Update pickaxe position (move it around as if mining)
        self._update_pickaxe_position()
        
        # Add sound effects
        if random.random() < 0.3:  # 30% chance each frame
            self._add_sound_effect()
        
        # Decay sound effects
        self._update_sound_effects()
        
        return self._render_frame()
    
    def _advance_mining(self):
        """Advance the mining progress, revealing content."""
        # Calculate how many cells should be revealed
        cells_to_reveal = int(self.mining_progress * self.total_cells)
        
        # Reveal cells in a mining pattern (roughly center-out)
        revealed_count = 0
        center_y = len(self.content_lines) // 2
        
        for distance in range(max(self.width, self.height)):
            if revealed_count >= cells_to_reveal:
                break
                
            # Mine in expanding pattern from center
            for line_idx, line in enumerate(self.content_lines):
                # Check line_idx bounds
                if line_idx >= self.height:
                    break
                if abs(line_idx - center_y) <= distance:
                    for char_idx, char in enumerate(line):
                        if revealed_count >= cells_to_reveal:
                            break
                        # Check bounds to prevent IndexError
                        if char_idx < self.width and not self.reveal_grid[line_idx][char_idx] and char != ' ':
                            self.reveal_grid[line_idx][char_idx] = True
                            # Set mining stage (gradually dig through stone)
                            self.mining_grid[line_idx][char_idx] = min(4, int(self.mining_progress * 8))
                            revealed_count += 1
    
    def _update_pickaxe_position(self):
        """Move pickaxe around as if actively mining."""
        # Simple circular motion with some randomness
        angle = time.time() * 2.0 + random.uniform(-0.5, 0.5)
        center_x = self.width // 2
        center_y = self.height // 2
        
        radius = 3 + random.uniform(-1, 1)
        self.pickaxe_pos[0] = int(center_x + math.cos(angle) * radius)
        self.pickaxe_pos[1] = int(center_y + math.sin(angle) * radius * 0.5)  # Smaller vertical movement
        
        # Keep within bounds
        self.pickaxe_pos[0] = max(0, min(self.width - 1, self.pickaxe_pos[0]))
        self.pickaxe_pos[1] = max(0, min(self.height - 1, self.pickaxe_pos[1]))
    
    def _add_sound_effect(self):
        """Add a sound effect near the pickaxe."""
        if len(self.sound_positions) < 5:  # Limit number of effects
            effect_char = random.choice(self.sound_effects)
            # Place near pickaxe with some randomness
            x = self.pickaxe_pos[0] + random.randint(-2, 2)
            y = self.pickaxe_pos[1] + random.randint(-1, 1)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                self.sound_positions.append({
                    'char': effect_char,
                    'x': x,
                    'y': y,
                    'life': 1.0,
                    'style': random.choice(['bold red', 'bold orange', 'bold yellow'])
                })
    
    def _update_sound_effects(self):
        """Update and decay sound effects."""
        for effect in self.sound_positions[:]:  # Copy list to modify while iterating
            effect['life'] -= 0.1
            if effect['life'] <= 0:
                self.sound_positions.remove(effect)
    
    def _render_frame(self) -> str:
        """Render the current mining frame."""
        # Start with empty grid
        display_grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [['' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw the stone/mining progress
        for y in range(min(len(self.content_lines), self.height)):
            line = self.content_lines[y]
            for x in range(min(len(line), self.width)):
                char = line[x]
                
                if char != ' ':
                    if self.reveal_grid[y][x]:
                        # Cell is being mined - show mining progress
                        stage = self.mining_grid[y][x]
                        if stage >= len(self.stone_chars) - 1:
                            # Fully mined - show original character
                            display_grid[y][x] = char
                            style_grid[y][x] = 'white'
                        else:
                            # Show mining progress
                            display_grid[y][x] = self.stone_chars[stage]
                            style_grid[y][x] = self.stone_styles[stage]
                    else:
                        # Unmined stone
                        display_grid[y][x] = self.stone_chars[0]
                        style_grid[y][x] = self.stone_styles[0]
        
        # Draw pickaxe
        px, py = self.pickaxe_pos
        if 0 <= px < self.width and 0 <= py < self.height:
            display_grid[py][px] = self.pickaxe_char
            style_grid[py][px] = self.pickaxe_style
        
        # Draw sound effects
        for effect in self.sound_positions:
            x, y = effect['x'], effect['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                if effect['life'] > 0.5:  # Only show if still fresh
                    display_grid[y][x] = effect['char']
                    style_grid[y][x] = effect['style']
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.height):
            line_segments = []
            for x in range(self.width):
                char = display_grid[y][x]
                style = style_grid[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class NeuralNetworkEffect(BaseEffect):
    """Neural network visualization with nodes and connections."""
    
    def __init__(self, parent, title="TLDW Chatbook", subtitle="", width=80, height=24, speed=0.1):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.subtitle = subtitle
        self.nodes = []
        self.connections = []
        self.activation_wave = 0
        
        # Create network structure
        layers = [3, 5, 4, 5, 3]  # Nodes per layer
        layer_spacing = self.width // (len(layers) + 1)
        
        for layer_idx, node_count in enumerate(layers):
            x = layer_spacing * (layer_idx + 1)
            layer_nodes = []
            node_spacing = self.height // (node_count + 1)
            
            for node_idx in range(node_count):
                y = node_spacing * (node_idx + 1)
                node = {
                    'x': x,
                    'y': y,
                    'layer': layer_idx,
                    'activation': 0.0,
                    'id': f"L{layer_idx}N{node_idx}"
                }
                layer_nodes.append(node)
                self.nodes.append(node)
            
            # Create connections to previous layer
            if layer_idx > 0:
                prev_layer_nodes = [n for n in self.nodes if n['layer'] == layer_idx - 1]
                for node in layer_nodes:
                    for prev_node in prev_layer_nodes:
                        if random.random() > 0.3:  # 70% connection probability
                            self.connections.append({
                                'from': prev_node,
                                'to': node,
                                'strength': random.random()
                            })
    
    def update(self, elapsed_time):
        """Update neural network animation."""
        self.activation_wave = (self.activation_wave + elapsed_time * 2) % (len(set(n['layer'] for n in self.nodes)) + 2)
        
        # Update node activations
        for node in self.nodes:
            if abs(node['layer'] - self.activation_wave) < 1:
                node['activation'] = min(1.0, node['activation'] + elapsed_time * 3)
            else:
                node['activation'] = max(0.0, node['activation'] - elapsed_time * 2)
    
    def render(self):
        """Render the neural network."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw connections
        for conn in self.connections:
            if conn['from']['activation'] > 0.1 or conn['to']['activation'] > 0.1:
                strength = (conn['from']['activation'] + conn['to']['activation']) / 2 * conn['strength']
                self._draw_line(grid, style_grid, 
                               conn['from']['x'], conn['from']['y'],
                               conn['to']['x'], conn['to']['y'],
                               strength)
        
        # Draw nodes
        for node in self.nodes:
            x, y = node['x'], node['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                if node['activation'] > 0.8:
                    grid[y][x] = '●'
                    style_grid[y][x] = 'bold cyan'
                elif node['activation'] > 0.4:
                    grid[y][x] = '◉'
                    style_grid[y][x] = 'cyan'
                elif node['activation'] > 0.1:
                    grid[y][x] = '○'
                    style_grid[y][x] = 'dim cyan'
                else:
                    grid[y][x] = '·'
                    style_grid[y][x] = 'dim white'
        
        # Add title when network is active
        if self.activation_wave > 1:
            self._add_centered_text(grid, style_grid, self.title, self.height // 2 - 1, 'bold white')
            if self.subtitle:
                self._add_centered_text(grid, style_grid, self.subtitle, self.height // 2 + 1, 'white')
        
        return self._grid_to_string(grid, style_grid)
    
    def _draw_line(self, grid, style_grid, x1, y1, x2, y2, strength):
        """Draw a line between two points."""
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return
        
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                if grid[y][x] == ' ':
                    if strength > 0.7:
                        grid[y][x] = '═' if abs(x2 - x1) > abs(y2 - y1) else '║'
                        style_grid[y][x] = 'bold blue'
                    elif strength > 0.3:
                        grid[y][x] = '─' if abs(x2 - x1) > abs(y2 - y1) else '│'
                        style_grid[y][x] = 'blue'
                    else:
                        grid[y][x] = '·'
                        style_grid[y][x] = 'dim blue'


class QuantumParticlesEffect(BaseEffect):
    """Quantum particles with superposition and entanglement effects."""
    
    def __init__(self, parent, title="TLDW Chatbook", subtitle="", width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.subtitle = subtitle
        self.particles = []
        self.entangled_pairs = []
        self.interference_pattern = [[0 for _ in range(width)] for _ in range(height)]
        
        # Create initial particles
        for _ in range(15):
            self.particles.append(self._create_particle())
        
        # Create entangled pairs
        for i in range(0, len(self.particles) - 1, 2):
            self.entangled_pairs.append((i, i + 1))
    
    def _create_particle(self):
        return {
            'x': random.uniform(0, self.width),
            'y': random.uniform(0, self.height),
            'vx': random.uniform(-0.5, 0.5),
            'vy': random.uniform(-0.5, 0.5),
            'phase': random.uniform(0, 2 * 3.14159),
            'superposition': random.choice([True, False]),
            'collapsed': False
        }
    
    def update(self, elapsed_time):
        """Update quantum particles."""
        # Update particles
        for i, particle in enumerate(self.particles):
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['phase'] += elapsed_time * 2
            
            # Bounce off walls
            if particle['x'] <= 0 or particle['x'] >= self.width - 1:
                particle['vx'] *= -1
            if particle['y'] <= 0 or particle['y'] >= self.height - 1:
                particle['vy'] *= -1
            
            # Random collapse/superposition
            if random.random() < 0.02:
                particle['superposition'] = not particle['superposition']
            
            # Update interference pattern
            x, y = int(particle['x']), int(particle['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                self.interference_pattern[y][x] = (self.interference_pattern[y][x] + 0.1) % 1.0
        
        # Entanglement effects
        for p1_idx, p2_idx in self.entangled_pairs:
            if random.random() < 0.1:
                # Quantum teleportation
                p1, p2 = self.particles[p1_idx], self.particles[p2_idx]
                p1['x'], p2['x'] = p2['x'], p1['x']
                p1['y'], p2['y'] = p2['y'], p1['y']
    
    def render(self):
        """Render quantum particles."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw interference pattern
        for y in range(self.height):
            for x in range(self.width):
                if self.interference_pattern[y][x] > 0.5:
                    grid[y][x] = '·'
                    style_grid[y][x] = 'dim cyan'
        
        # Draw entanglement lines
        for p1_idx, p2_idx in self.entangled_pairs:
            p1, p2 = self.particles[p1_idx], self.particles[p2_idx]
            if p1['superposition'] and p2['superposition']:
                self._draw_quantum_line(grid, style_grid,
                                      int(p1['x']), int(p1['y']),
                                      int(p2['x']), int(p2['y']))
        
        # Draw particles
        for particle in self.particles:
            x, y = int(particle['x']), int(particle['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                if particle['superposition']:
                    grid[y][x] = '◈'
                    style_grid[y][x] = 'bold magenta'
                else:
                    grid[y][x] = '●'
                    style_grid[y][x] = 'cyan'
        
        # Add title
        self._add_centered_text(grid, style_grid, self.title, self.height // 2 - 1, 'bold white')
        if self.subtitle:
            self._add_centered_text(grid, style_grid, self.subtitle, self.height // 2 + 1, 'white')
        
        return self._grid_to_string(grid, style_grid)
    
    def _draw_quantum_line(self, grid, style_grid, x1, y1, x2, y2):
        """Draw a quantum entanglement line."""
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return
        
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                if grid[y][x] == ' ' or grid[y][x] == '·':
                    grid[y][x] = '~'
                    style_grid[y][x] = 'dim magenta'


class ASCIIWaveEffect(BaseEffect):
    """Ocean waves animation with ASCII characters."""
    
    def __init__(self, parent, title="TLDW Chatbook", subtitle="", width=80, height=24, speed=0.1):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.subtitle = subtitle
        self.wave_offset = 0
        self.wave_chars = ['_', '-', '~', '≈', '~', '-', '_']
        self.foam_chars = ['·', '°', '*', '°', '·']
    
    def update(self, elapsed_time):
        """Update wave animation."""
        self.wave_offset += elapsed_time * 5
    
    def render(self):
        """Render ocean waves."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate wave heights
        for x in range(self.width):
            # Primary wave
            wave1 = math.sin((x + self.wave_offset) * 0.1) * 3
            wave2 = math.sin((x + self.wave_offset * 0.7) * 0.15) * 2
            wave3 = math.sin((x + self.wave_offset * 1.3) * 0.05) * 4
            
            total_wave = wave1 + wave2 + wave3
            wave_height = int(self.height / 2 + total_wave)
            
            # Draw water column
            for y in range(self.height):
                if y > wave_height:
                    # Below water
                    depth = y - wave_height
                    if depth < len(self.wave_chars):
                        grid[y][x] = self.wave_chars[depth]
                        intensity = 255 - depth * 20
                        style_grid[y][x] = f'rgb({intensity//2},{intensity//2},{intensity})'
                    else:
                        grid[y][x] = '▓'
                        style_grid[y][x] = 'blue'
                elif y == wave_height:
                    # Wave crest
                    if random.random() < 0.3:
                        grid[y][x] = random.choice(self.foam_chars)
                        style_grid[y][x] = 'bold white'
                    else:
                        grid[y][x] = '≈'
                        style_grid[y][x] = 'bold cyan'
        
        # Add title in the sky
        self._add_centered_text(grid, style_grid, self.title, self.height // 4, 'bold white')
        if self.subtitle:
            self._add_centered_text(grid, style_grid, self.subtitle, self.height // 4 + 2, 'white')
        
        return self._grid_to_string(grid, style_grid)


class BinaryMatrixEffect(BaseEffect):
    """Binary rain effect with highlighting patterns."""
    
    def __init__(self, parent, title="TLDW", width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.columns = []
        self.highlight_pattern = "TLDW"
        self.highlight_positions = []
        
        # Initialize columns
        for x in range(self.width):
            self.columns.append({
                'chars': ['0', '1'] * self.height,
                'offset': random.randint(0, self.height),
                'speed': random.uniform(0.5, 2.0),
                'highlight': False
            })
    
    def update(self, elapsed_time):
        """Update binary rain."""
        for col in self.columns:
            col['offset'] += col['speed']
            if col['offset'] >= self.height * 2:
                col['offset'] = 0
                col['speed'] = random.uniform(0.5, 2.0)
                # Randomly generate new binary sequence
                col['chars'] = [random.choice(['0', '1']) for _ in range(self.height * 2)]
        
        # Update highlight positions
        if random.random() < 0.05:
            self._create_highlight()
    
    def _create_highlight(self):
        """Create a highlighted pattern in the binary rain."""
        start_x = random.randint(0, self.width - len(self.highlight_pattern))
        start_y = random.randint(0, self.height - 1)
        
        self.highlight_positions = []
        for i, char in enumerate(self.highlight_pattern):
            self.highlight_positions.append({
                'x': start_x + i,
                'y': start_y,
                'char': char,
                'life': 2.0
            })
    
    def render(self):
        """Render binary matrix rain."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw binary columns
        for x, col in enumerate(self.columns):
            offset = int(col['offset'])
            for y in range(self.height):
                char_idx = (y + offset) % len(col['chars'])
                char = col['chars'][char_idx]
                
                # Fade based on position
                distance_from_head = (offset - y) % self.height
                if distance_from_head < 3:
                    style = 'bold green'
                elif distance_from_head < 10:
                    style = 'green'
                else:
                    style = 'dim green'
                
                grid[y][x] = char
                style_grid[y][x] = style
        
        # Draw highlights
        for highlight in self.highlight_positions[:]:
            if highlight['life'] > 0:
                x, y = highlight['x'], highlight['y']
                if 0 <= x < self.width and 0 <= y < self.height:
                    grid[y][x] = highlight['char']
                    style_grid[y][x] = 'bold yellow' if highlight['life'] > 1 else 'yellow'
                highlight['life'] -= 0.05
            else:
                self.highlight_positions.remove(highlight)
        
        return self._grid_to_string(grid, style_grid)


class ConstellationMapEffect(BaseEffect):
    """Stars forming constellations that spell out the logo."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.1):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.stars = []
        self.connections = []
        self.shooting_stars = []
        self.reveal_progress = 0
        
        # Create background stars
        for _ in range(50):
            self.stars.append({
                'x': random.randint(0, width - 1),
                'y': random.randint(0, height - 1),
                'brightness': random.choice(['·', '•', '*']),
                'twinkle': random.random(),
                'constellation': False
            })
        
        # Create constellation pattern (simplified TLDW shape)
        self._create_constellation()
    
    def _create_constellation(self):
        """Create constellation in shape of letters."""
        # T constellation
        cx = self.width // 4
        cy = self.height // 2
        t_stars = [
            (cx - 4, cy - 4), (cx, cy - 4), (cx + 4, cy - 4),  # Top of T
            (cx, cy - 2), (cx, cy), (cx, cy + 2), (cx, cy + 4)  # Stem of T
        ]
        
        # L constellation
        cx = self.width // 2
        l_stars = [
            (cx - 4, cy - 4), (cx - 4, cy - 2), (cx - 4, cy), (cx - 4, cy + 2), (cx - 4, cy + 4),  # Vertical
            (cx - 2, cy + 4), (cx, cy + 4), (cx + 2, cy + 4)  # Horizontal
        ]
        
        # Add constellation stars
        for x, y in t_stars + l_stars:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.stars.append({
                    'x': x,
                    'y': y,
                    'brightness': '★',
                    'twinkle': 0,
                    'constellation': True,
                    'revealed': False
                })
        
        # Create connections for T
        for i in range(3):  # Top horizontal line
            self.connections.append((t_stars[i], t_stars[i + 1] if i < 2 else t_stars[1]))
        for i in range(len(t_stars) - 4):  # Vertical line
            if i + 4 < len(t_stars):
                self.connections.append((t_stars[i + 3], t_stars[i + 4]))
        
        # Create connections for L
        for i in range(4):  # Vertical line
            if i + 1 < 5:
                self.connections.append((l_stars[i], l_stars[i + 1]))
        for i in range(5, 7):  # Horizontal line
            if i + 1 < len(l_stars):
                self.connections.append((l_stars[i], l_stars[i + 1]))
    
    def update(self, elapsed_time):
        """Update constellation animation."""
        # Twinkle stars
        for star in self.stars:
            if not star['constellation']:
                star['twinkle'] = (star['twinkle'] + elapsed_time * random.uniform(1, 3)) % 1.0
        
        # Reveal constellation gradually
        self.reveal_progress += elapsed_time * 0.5
        revealed_count = int(self.reveal_progress * len([s for s in self.stars if s['constellation']]))
        
        constellation_stars = [s for s in self.stars if s['constellation']]
        for i, star in enumerate(constellation_stars):
            if i < revealed_count:
                star['revealed'] = True
        
        # Create shooting stars
        if random.random() < 0.02:
            self.shooting_stars.append({
                'x': random.randint(0, self.width),
                'y': 0,
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(0.5, 1.5),
                'trail': []
            })
        
        # Update shooting stars
        for star in self.shooting_stars[:]:
            star['x'] += star['vx']
            star['y'] += star['vy']
            star['trail'].append((int(star['x']), int(star['y'])))
            if len(star['trail']) > 5:
                star['trail'].pop(0)
            
            if star['y'] >= self.height:
                self.shooting_stars.remove(star)
    
    def render(self):
        """Render constellation map."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw background stars
        for star in self.stars:
            x, y = int(star['x']), int(star['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                if star['constellation'] and star.get('revealed', False):
                    grid[y][x] = star['brightness']
                    style_grid[y][x] = 'bold yellow'
                elif not star['constellation']:
                    if star['twinkle'] > 0.7:
                        grid[y][x] = star['brightness']
                        style_grid[y][x] = 'white'
                    else:
                        grid[y][x] = '·'
                        style_grid[y][x] = 'dim white'
        
        # Draw constellation connections
        if self.reveal_progress > 0.3:
            for (x1, y1), (x2, y2) in self.connections:
                star1 = next((s for s in self.stars if s['x'] == x1 and s['y'] == y1 and s.get('revealed')), None)
                star2 = next((s for s in self.stars if s['x'] == x2 and s['y'] == y2 and s.get('revealed')), None)
                if star1 and star2:
                    self._draw_constellation_line(grid, style_grid, x1, y1, x2, y2)
        
        # Draw shooting stars
        for star in self.shooting_stars:
            for i, (x, y) in enumerate(star['trail']):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if i == len(star['trail']) - 1:
                        grid[y][x] = '✦'
                        style_grid[y][x] = 'bold white'
                    else:
                        grid[y][x] = '·'
                        style_grid[y][x] = 'dim white'
        
        # Add title when constellation is mostly revealed
        if self.reveal_progress > 0.7:
            self._add_centered_text(grid, style_grid, self.title, self.height - 3, 'bold white')
        
        return self._grid_to_string(grid, style_grid)
    
    def _draw_constellation_line(self, grid, style_grid, x1, y1, x2, y2):
        """Draw a faint line between constellation stars."""
        steps = max(abs(x2 - x1), abs(y2 - y1)) * 2
        if steps == 0:
            return
        
        for i in range(1, steps):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                if grid[y][x] == ' ':
                    grid[y][x] = '·'
                    style_grid[y][x] = 'dim yellow'


class TypewriterNewsEffect(BaseEffect):
    """Old newspaper typewriter effect with breaking news."""
    
    def __init__(self, parent, width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.typed_chars = 0
        self.paper_lines = []
        self.cursor_blink = 0
        self.carriage_return_sound = False
        
        # News content
        self.headline = "BREAKING: TLDW CHATBOOK LAUNCHES!"
        self.subheadline = "Revolutionary AI Assistant Takes Terminal By Storm"
        self.dateline = "Terminal City - " + time.strftime("%B %d, %Y")
        self.article = [
            "In a stunning development today, the highly anticipated",
            "TLDW Chatbook has been released to the public. This",
            "groundbreaking terminal-based AI assistant promises to",
            "revolutionize how users interact with language models.",
            "",
            "Early reports indicate unprecedented user satisfaction",
            "with the innovative ASCII-based interface and powerful",
            "conversation management features.",
            "",
            '"This changes everything," said one beta tester.',
        ]
    
    def update(self, elapsed_time):
        """Update typewriter animation."""
        # Type characters
        self.typed_chars += elapsed_time * 30  # Characters per second
        
        # Cursor blink
        self.cursor_blink = (self.cursor_blink + elapsed_time * 3) % 1.0
        
        # Check for carriage return
        if int(self.typed_chars) > 0 and int(self.typed_chars) % 60 == 0:
            self.carriage_return_sound = True
        else:
            self.carriage_return_sound = False
    
    def render(self):
        """Render typewriter news effect."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw paper background
        for y in range(2, self.height - 2):
            for x in range(5, self.width - 5):
                grid[y][x] = ' '
                style_grid[y][x] = 'on rgb(245,245,220)'  # Light beige
        
        # Draw paper edges
        for y in range(2, self.height - 2):
            grid[y][4] = '│'
            grid[y][self.width - 5] = '│'
            style_grid[y][4] = style_grid[y][self.width - 5] = 'black'
        
        # Type content
        current_line = 4
        chars_typed = int(self.typed_chars)
        
        # Type headline
        if chars_typed > 0:
            headline_typed = self.headline[:min(chars_typed, len(self.headline))]
            self._add_centered_text(grid, style_grid, headline_typed, current_line, 'bold black on rgb(245,245,220)')
            chars_typed -= len(self.headline)
            current_line += 2
        
        # Type subheadline
        if chars_typed > 0:
            subheadline_typed = self.subheadline[:min(chars_typed, len(self.subheadline))]
            self._add_centered_text(grid, style_grid, subheadline_typed, current_line, 'black on rgb(245,245,220)')
            chars_typed -= len(self.subheadline)
            current_line += 2
        
        # Type dateline
        if chars_typed > 0:
            dateline_typed = self.dateline[:min(chars_typed, len(self.dateline))]
            self._add_text_at(grid, style_grid, dateline_typed, 8, current_line, 'italic black on rgb(245,245,220)')
            chars_typed -= len(self.dateline)
            current_line += 2
        
        # Type article
        for line in self.article:
            if chars_typed > 0 and current_line < self.height - 4:
                line_typed = line[:min(chars_typed, len(line))]
                self._add_text_at(grid, style_grid, line_typed, 8, current_line, 'black on rgb(245,245,220)')
                chars_typed -= len(line)
                current_line += 1
        
        # Draw cursor
        if self.cursor_blink > 0.5 and chars_typed >= 0:
            cursor_pos = min(int(self.typed_chars), sum(len(line) for line in [self.headline, self.subheadline, self.dateline] + self.article))
            # Find cursor position (simplified)
            if current_line < self.height - 4:
                grid[current_line][min(8 + (cursor_pos % 50), self.width - 6)] = '█'
                style_grid[current_line][min(8 + (cursor_pos % 50), self.width - 6)] = 'black on rgb(245,245,220)'
        
        # Add typewriter sound effect
        if self.carriage_return_sound:
            self._add_text_at(grid, style_grid, "DING!", 2, 2, 'bold red')
        
        return self._grid_to_string(grid, style_grid)
    
    def _add_text_at(self, grid, style_grid, text, x, y, style):
        """Add text at specific position."""
        for i, char in enumerate(text):
            if x + i < len(grid[0]):
                grid[y][x + i] = char
                style_grid[y][x + i] = style


class DNASequenceEffect(BaseEffect):
    """Enhanced DNA double helix with genetic code."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.rotation = 0
        self.base_pairs = ['A-T', 'T-A', 'G-C', 'C-G']
        self.mutation_chance = 0.01
        self.gene_sequence = "INTELLIGENCEAUGMENTED"
        self.reveal_progress = 0
    
    def update(self, elapsed_time):
        """Update DNA rotation and mutations."""
        self.rotation += elapsed_time * 1.5
        self.reveal_progress = min(1.0, self.reveal_progress + elapsed_time * 0.3)
        
        # Random mutations
        if random.random() < self.mutation_chance:
            self.mutation_flash = 1.0
    
    def render(self):
        """Render DNA double helix."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        center_x = self.width // 2
        helix_width = 20
        
        for y in range(self.height):
            # Calculate helix position
            angle = (y * 0.5 + self.rotation) % (2 * math.pi)
            left_offset = int(math.sin(angle) * helix_width)
            right_offset = int(math.sin(angle + math.pi) * helix_width)
            
            left_x = center_x + left_offset
            right_x = center_x + right_offset
            
            # Draw backbone
            if 0 <= left_x < self.width:
                grid[y][left_x] = '|'
                style_grid[y][left_x] = 'bold blue'
            
            if 0 <= right_x < self.width:
                grid[y][right_x] = '|'
                style_grid[y][right_x] = 'bold blue'
            
            # Draw base pairs when strands cross
            if abs(left_offset - right_offset) < 3:
                base_pair = random.choice(self.base_pairs)
                connection_start = min(left_x, right_x) + 1
                connection_end = max(left_x, right_x)
                
                if connection_end - connection_start > 2:
                    mid = (connection_start + connection_end) // 2
                    if 0 <= mid - 1 < self.width and 0 <= mid + 1 < self.width:
                        grid[y][mid - 1] = base_pair[0]
                        grid[y][mid] = '-'
                        grid[y][mid + 1] = base_pair[2]
                        
                        # Color based on base
                        color_map = {'A': 'red', 'T': 'green', 'G': 'yellow', 'C': 'cyan'}
                        style_grid[y][mid - 1] = color_map.get(base_pair[0], 'white')
                        style_grid[y][mid] = 'white'
                        style_grid[y][mid + 1] = color_map.get(base_pair[2], 'white')
        
        # Add gene sequence reveal
        if self.reveal_progress > 0.3:
            seq_y = self.height // 2
            seq_x = (self.width - len(self.gene_sequence)) // 2
            revealed_chars = int(self.reveal_progress * len(self.gene_sequence))
            
            for i in range(revealed_chars):
                if seq_x + i < self.width:
                    grid[seq_y][seq_x + i] = self.gene_sequence[i]
                    style_grid[seq_y][seq_x + i] = 'bold white'
        
        # Add title
        if self.reveal_progress > 0.7:
            self._add_centered_text(grid, style_grid, self.title, 2, 'bold white')
        
        return self._grid_to_string(grid, style_grid)


class CircuitTraceEffect(BaseEffect):
    """PCB circuit traces being drawn with electrical signals."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.02):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.traces = []
        self.components = []
        self.signals = []
        self.trace_progress = 0
        
        # Generate circuit layout
        self._generate_circuit()
    
    def _generate_circuit(self):
        """Generate a circuit board layout."""
        # Main bus lines
        for y in [5, 10, 15, 20]:
            self.traces.append({
                'start': (5, y),
                'end': (self.width - 5, y),
                'drawn': 0,
                'type': 'horizontal'
            })
        
        # Vertical connections
        for x in range(10, self.width - 10, 15):
            y_start = random.choice([5, 10])
            y_end = random.choice([15, 20])
            self.traces.append({
                'start': (x, y_start),
                'end': (x, y_end),
                'drawn': 0,
                'type': 'vertical'
            })
        
        # Add components
        component_types = [
            {'symbol': '[R]', 'name': 'resistor'},
            {'symbol': '[C]', 'name': 'capacitor'},
            {'symbol': '[D]', 'name': 'diode'},
            {'symbol': '[U]', 'name': 'chip'}
        ]
        
        for _ in range(10):
            self.components.append({
                'x': random.randint(10, self.width - 10),
                'y': random.choice([5, 10, 15, 20]),
                'type': random.choice(component_types),
                'placed': False
            })
    
    def update(self, elapsed_time):
        """Update circuit trace animation."""
        self.trace_progress = min(1.0, self.trace_progress + elapsed_time * 0.5)
        
        # Update trace drawing
        for trace in self.traces:
            trace['drawn'] = min(1.0, trace['drawn'] + elapsed_time * 2)
        
        # Create electrical signals
        if random.random() < 0.1 and self.trace_progress > 0.3:
            trace = random.choice([t for t in self.traces if t['drawn'] > 0.5])
            self.signals.append({
                'trace': trace,
                'position': 0,
                'speed': random.uniform(20, 40)
            })
        
        # Update signals
        for signal in self.signals[:]:
            signal['position'] += signal['speed'] * elapsed_time
            if signal['position'] > 1.0:
                self.signals.remove(signal)
        
        # Place components
        for comp in self.components:
            if not comp['placed'] and self.trace_progress > random.uniform(0.2, 0.8):
                comp['placed'] = True
    
    def render(self):
        """Render circuit board."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw PCB background
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < 0.02:
                    grid[y][x] = '·'
                    style_grid[y][x] = 'dim green'
        
        # Draw traces
        for trace in self.traces:
            x1, y1 = trace['start']
            x2, y2 = trace['end']
            length = max(abs(x2 - x1), abs(y2 - y1))
            drawn_length = int(length * trace['drawn'])
            
            if trace['type'] == 'horizontal':
                for i in range(drawn_length + 1):
                    x = x1 + i if x2 > x1 else x1 - i
                    if 0 <= x < self.width:
                        grid[y1][x] = '═'
                        style_grid[y1][x] = 'yellow'
            else:  # vertical
                for i in range(drawn_length + 1):
                    y = y1 + i if y2 > y1 else y1 - i
                    if 0 <= y < self.height:
                        grid[y][x1] = '║'
                        style_grid[y][x1] = 'yellow'
        
        # Draw components
        for comp in self.components:
            if comp['placed']:
                x, y = comp['x'], comp['y']
                symbol = comp['type']['symbol']
                if x + len(symbol) < self.width and 0 <= y < self.height:
                    for i, char in enumerate(symbol):
                        grid[y][x + i] = char
                        style_grid[y][x + i] = 'bold cyan'
        
        # Draw signals
        for signal in self.signals:
            trace = signal['trace']
            x1, y1 = trace['start']
            x2, y2 = trace['end']
            
            # Calculate signal position
            t = signal['position']
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = '●'
                style_grid[y][x] = 'bold white'
        
        # Add title when circuit is mostly complete
        if self.trace_progress > 0.7:
            self._add_centered_text(grid, style_grid, self.title, self.height // 2, 'bold white')
        
        return self._grid_to_string(grid, style_grid)


class PlasmaFieldEffect(BaseEffect):
    """Animated plasma field effect."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.time_offset = 0
        self.plasma_chars = [' ', '·', ':', '░', '▒', '▓', '█']
    
    def update(self, elapsed_time):
        """Update plasma field."""
        self.time_offset += elapsed_time * 2
    
    def render(self):
        """Render plasma field."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Generate plasma field
        for y in range(self.height):
            for x in range(self.width):
                # Calculate plasma value using multiple sine waves
                v1 = math.sin((x * 0.1) + self.time_offset)
                v2 = math.sin((y * 0.1) + self.time_offset * 1.3)
                v3 = math.sin(((x + y) * 0.05) + self.time_offset * 0.7)
                v4 = math.sin(math.sqrt((x - self.width/2)**2 + (y - self.height/2)**2) * 0.1 - self.time_offset)
                
                # Combine waves
                plasma_value = (v1 + v2 + v3 + v4) / 4
                normalized = (plasma_value + 1) / 2  # Normalize to 0-1
                
                # Select character based on plasma value
                char_index = int(normalized * (len(self.plasma_chars) - 1))
                grid[y][x] = self.plasma_chars[char_index]
                
                # Color based on plasma value
                if normalized < 0.25:
                    style_grid[y][x] = 'blue'
                elif normalized < 0.5:
                    style_grid[y][x] = 'cyan'
                elif normalized < 0.75:
                    style_grid[y][x] = 'magenta'
                else:
                    style_grid[y][x] = 'red'
        
        # Clear area for title
        title_y = self.height // 2
        title_area_height = 3
        for y in range(title_y - 1, title_y + title_area_height - 1):
            for x in range(self.width // 4, 3 * self.width // 4):
                if 0 <= y < self.height:
                    grid[y][x] = ' '
                    style_grid[y][x] = None
        
        # Add title
        self._add_centered_text(grid, style_grid, self.title, title_y, 'bold white')
        
        return self._grid_to_string(grid, style_grid)


class ASCIIFireEffect(BaseEffect):
    """Realistic fire animation using ASCII characters."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.fire_chars = [' ', '.', ':', '^', '*', '†', '‡', '¥', '§']
        self.fire_grid = [[0 for _ in range(width)] for _ in range(height)]
        self.embers = []
    
    def update(self, elapsed_time):
        """Update fire animation."""
        # Add new fire at bottom
        for x in range(self.width):
            if random.random() < 0.8:
                intensity = random.randint(6, 8)
                self.fire_grid[self.height - 1][x] = intensity
        
        # Propagate fire upwards
        new_grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        for y in range(self.height - 1):
            for x in range(self.width):
                # Get fire from below with some spreading
                below = self.fire_grid[y + 1][x]
                left = self.fire_grid[y + 1][x - 1] if x > 0 else 0
                right = self.fire_grid[y + 1][x + 1] if x < self.width - 1 else 0
                
                # Average with decay
                avg = (below * 0.97 + left * 0.01 + right * 0.01)
                new_grid[y][x] = max(0, avg - random.uniform(0, 0.5))
        
        # Copy bottom row
        new_grid[self.height - 1] = self.fire_grid[self.height - 1][:]
        self.fire_grid = new_grid
        
        # Create embers
        if random.random() < 0.1:
            self.embers.append({
                'x': random.randint(self.width // 3, 2 * self.width // 3),
                'y': self.height - 5,
                'vy': -random.uniform(0.5, 1.5),
                'life': 1.0
            })
        
        # Update embers
        for ember in self.embers[:]:
            ember['y'] += ember['vy']
            ember['vy'] += 0.1  # Gravity
            ember['life'] -= elapsed_time * 0.5
            
            if ember['life'] <= 0 or ember['y'] >= self.height:
                self.embers.remove(ember)
    
    def render(self):
        """Render fire effect."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw fire
        for y in range(self.height):
            for x in range(self.width):
                intensity = self.fire_grid[y][x]
                if intensity > 0:
                    char_index = min(int(intensity), len(self.fire_chars) - 1)
                    grid[y][x] = self.fire_chars[char_index]
                    
                    # Color based on intensity
                    if intensity > 6:
                        style_grid[y][x] = 'bold white'
                    elif intensity > 4:
                        style_grid[y][x] = 'bold yellow'
                    elif intensity > 2:
                        style_grid[y][x] = 'red'
                    else:
                        style_grid[y][x] = 'dim red'
        
        # Draw embers
        for ember in self.embers:
            x, y = int(ember['x']), int(ember['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = '°'
                style_grid[y][x] = 'yellow' if ember['life'] > 0.5 else 'dim red'
        
        # Add title in the flames
        title_y = self.height // 3
        self._add_centered_text(grid, style_grid, self.title, title_y, 'bold white on red')
        
        return self._grid_to_string(grid, style_grid)


class RubiksCubeEffect(BaseEffect):
    """3D ASCII Rubik's cube solving itself."""
    
    def __init__(self, parent, title="TLDW", width=80, height=24, speed=0.5):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.rotation_x = 0
        self.rotation_y = 0
        self.solve_progress = 0
        self.current_move = None
        self.move_progress = 0
        
        # Simplified cube representation
        self.faces = {
            'F': [['R' for _ in range(3)] for _ in range(3)],  # Front - Red
            'B': [['O' for _ in range(3)] for _ in range(3)],  # Back - Orange
            'U': [['W' for _ in range(3)] for _ in range(3)],  # Up - White
            'D': [['Y' for _ in range(3)] for _ in range(3)],  # Down - Yellow
            'L': [['G' for _ in range(3)] for _ in range(3)],  # Left - Green
            'R': [['B' for _ in range(3)] for _ in range(3)]   # Right - Blue
        }
        
        # Scramble cube
        self._scramble()
    
    def _scramble(self):
        """Scramble the cube."""
        moves = ['F', 'B', 'U', 'D', 'L', 'R']
        for _ in range(20):
            self._rotate_face(random.choice(moves))
    
    def _rotate_face(self, face):
        """Rotate a face 90 degrees clockwise."""
        # Simplified rotation - just shuffle colors
        if face in self.faces:
            # Rotate the face itself
            face_data = self.faces[face]
            rotated = [[face_data[2-j][i] for j in range(3)] for i in range(3)]
            self.faces[face] = rotated
    
    def update(self, elapsed_time):
        """Update cube animation."""
        # Rotate cube for 3D effect
        self.rotation_y += elapsed_time * 0.5
        
        # Solve animation
        self.solve_progress += elapsed_time * 0.1
        
        # Perform moves
        if not self.current_move and random.random() < 0.3:
            self.current_move = random.choice(['F', 'B', 'U', 'D', 'L', 'R'])
            self.move_progress = 0
        
        if self.current_move:
            self.move_progress += elapsed_time * 3
            if self.move_progress >= 1.0:
                self._rotate_face(self.current_move)
                self.current_move = None
    
    def render(self):
        """Render 3D Rubik's cube."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Draw isometric cube
        # Top face (U)
        top_start_x = center_x - 6
        top_start_y = center_y - 8
        
        for i in range(3):
            for j in range(3):
                x = top_start_x + i * 4 + j * 2
                y = top_start_y + i * 2 - j
                if 0 <= x < self.width and 0 <= y < self.height:
                    color = self.faces['U'][i][j]
                    grid[y][x] = '▄'
                    grid[y][x + 1] = '▄'
                    style_grid[y][x] = style_grid[y][x + 1] = self._get_color_style(color)
        
        # Front face (F)
        front_start_x = center_x - 6
        front_start_y = center_y - 2
        
        for i in range(3):
            for j in range(3):
                x = front_start_x + j * 4
                y = front_start_y + i * 2
                if 0 <= x < self.width and 0 <= y < self.height:
                    color = self.faces['F'][i][j]
                    grid[y][x] = '█'
                    grid[y][x + 1] = '█'
                    if y + 1 < self.height:
                        grid[y + 1][x] = '█'
                        grid[y + 1][x + 1] = '█'
                    style_grid[y][x] = style_grid[y][x + 1] = self._get_color_style(color)
                    if y + 1 < self.height:
                        style_grid[y + 1][x] = style_grid[y + 1][x + 1] = self._get_color_style(color)
        
        # Right face (R) - partial view
        right_start_x = center_x + 6
        right_start_y = center_y - 5
        
        for i in range(3):
            for j in range(3):
                x = right_start_x + j * 2
                y = right_start_y + i * 2 + j
                if 0 <= x < self.width and 0 <= y < self.height:
                    color = self.faces['R'][i][j]
                    grid[y][x] = '▐'
                    style_grid[y][x] = self._get_color_style(color)
        
        # Add title when cube is solving
        if self.solve_progress > 0.3:
            self._add_centered_text(grid, style_grid, self.title, self.height - 3, 'bold white')
            if self.solve_progress > 0.7:
                self._add_centered_text(grid, style_grid, "SOLVED!", self.height - 1, 'bold green')
        
        return self._grid_to_string(grid, style_grid)
    
    def _get_color_style(self, color_char):
        """Get style for cube colors."""
        color_map = {
            'R': 'red',
            'O': 'rgb(255,165,0)',  # Orange
            'W': 'white',
            'Y': 'yellow',
            'G': 'green',
            'B': 'blue'
        }
        return f"bold {color_map.get(color_char, 'white')}"


class DataStreamEffect(BaseEffect):
    """Hexadecimal data streaming with hidden messages."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.02):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.data_lines = []
        self.decoded_message = "TERMINAL LANGUAGE DATA WATCHER"
        self.decode_progress = 0
        self.highlight_positions = []
        
        # Initialize data lines
        for _ in range(height):
            self.data_lines.append(self._generate_data_line())
    
    def _generate_data_line(self):
        """Generate a line of hex data."""
        hex_chars = '0123456789ABCDEF'
        line = []
        for _ in range(self.width // 3):
            line.append(random.choice(hex_chars) + random.choice(hex_chars))
        return line
    
    def update(self, elapsed_time):
        """Update data stream."""
        # Scroll data
        if random.random() < 0.3:
            self.data_lines.pop(0)
            self.data_lines.append(self._generate_data_line())
        
        # Update decode progress
        self.decode_progress += elapsed_time * 0.2
        
        # Create highlight positions for decoded message
        if self.decode_progress > 0.3 and len(self.highlight_positions) < len(self.decoded_message):
            if random.random() < 0.1:
                self.highlight_positions.append({
                    'char': self.decoded_message[len(self.highlight_positions)],
                    'x': random.randint(0, self.width - 3),
                    'y': random.randint(0, self.height - 1)
                })
    
    def render(self):
        """Render data stream."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw hex data
        for y, line in enumerate(self.data_lines):
            x = 0
            for hex_pair in line:
                if x + 2 < self.width:
                    grid[y][x] = hex_pair[0]
                    grid[y][x + 1] = hex_pair[1]
                    
                    # Random highlighting
                    if random.random() < 0.05:
                        style_grid[y][x] = style_grid[y][x + 1] = 'bold green'
                    else:
                        style_grid[y][x] = style_grid[y][x + 1] = 'dim cyan'
                    
                    x += 3  # Space between hex pairs
        
        # Draw decoded characters
        for pos in self.highlight_positions:
            x, y = pos['x'], pos['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = pos['char']
                style_grid[y][x] = 'bold yellow'
        
        # Show full decoded message when complete
        if len(self.highlight_positions) >= len(self.decoded_message):
            msg_y = self.height // 2
            self._add_centered_text(grid, style_grid, self.decoded_message, msg_y - 1, 'bold white on green')
            self._add_centered_text(grid, style_grid, self.title, msg_y + 1, 'bold white')
        
        return self._grid_to_string(grid, style_grid)


class FractalZoomEffect(BaseEffect):
    """Mandelbrot fractal zoom effect."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.zoom = 1.0
        self.center_x = -0.5
        self.center_y = 0.0
        self.max_iter = 50
        self.chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    
    def update(self, elapsed_time):
        """Update fractal zoom."""
        self.zoom *= 1.0 + elapsed_time * 0.5
        
        # Slowly drift center
        self.center_x += elapsed_time * 0.01
    
    def _mandelbrot(self, c_real, c_imag):
        """Calculate Mandelbrot iteration count."""
        z_real, z_imag = 0, 0
        
        for i in range(self.max_iter):
            if z_real * z_real + z_imag * z_imag > 4:
                return i
            
            z_real_new = z_real * z_real - z_imag * z_imag + c_real
            z_imag = 2 * z_real * z_imag + c_imag
            z_real = z_real_new
        
        return self.max_iter
    
    def render(self):
        """Render fractal zoom."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate fractal
        for y in range(self.height):
            for x in range(self.width):
                # Map pixel to complex plane
                real = (x - self.width / 2) / (self.zoom * self.width / 4) + self.center_x
                imag = (y - self.height / 2) / (self.zoom * self.height / 4) + self.center_y
                
                # Calculate iterations
                iterations = self._mandelbrot(real, imag)
                
                # Map to character
                if iterations == self.max_iter:
                    grid[y][x] = ' '
                else:
                    char_index = iterations % len(self.chars)
                    grid[y][x] = self.chars[char_index]
                    
                    # Color based on iteration count
                    if iterations < 10:
                        style_grid[y][x] = 'blue'
                    elif iterations < 20:
                        style_grid[y][x] = 'cyan'
                    elif iterations < 30:
                        style_grid[y][x] = 'green'
                    elif iterations < 40:
                        style_grid[y][x] = 'yellow'
                    else:
                        style_grid[y][x] = 'red'
        
        # Add title overlay
        if self.zoom > 3:
            self._add_centered_text(grid, style_grid, self.title, self.height // 2, 'bold white on black')
        
        return self._grid_to_string(grid, style_grid)


class ASCIISpinnerEffect(BaseEffect):
    """Multiple synchronized loading spinners."""
    
    def __init__(self, parent, title="Loading TLDW Chatbook", width=80, height=24, speed=0.1):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.spinners = []
        self.phase = 0
        
        # Define spinner types
        self.spinner_types = [
            {'frames': ['|', '/', '-', '\\'], 'name': 'classic'},
            {'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'], 'name': 'braille'},
            {'frames': ['◐', '◓', '◑', '◒'], 'name': 'circle'},
            {'frames': ['◰', '◳', '◲', '◱'], 'name': 'square'},
            {'frames': ['▖', '▘', '▝', '▗'], 'name': 'dots'},
            {'frames': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'], 'name': 'arrows'},
            {'frames': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'], 'name': 'bars'}
        ]
        
        # Create spinner grid
        spacing_x = self.width // 4
        spacing_y = self.height // 4
        
        for i in range(3):
            for j in range(3):
                if i * 3 + j < len(self.spinner_types):
                    self.spinners.append({
                        'x': spacing_x * (j + 1),
                        'y': spacing_y * (i + 1),
                        'type': self.spinner_types[i * 3 + j],
                        'phase_offset': random.uniform(0, 1)
                    })
    
    def update(self, elapsed_time):
        """Update spinner animations."""
        self.phase += elapsed_time * 2
    
    def render(self):
        """Render multiple spinners."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw spinners
        for spinner in self.spinners:
            x, y = spinner['x'], spinner['y']
            frames = spinner['type']['frames']
            
            # Calculate frame index with phase offset
            frame_index = int((self.phase + spinner['phase_offset'] * len(frames)) % len(frames))
            
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = frames[frame_index]
                
                # Color based on spinner type
                if spinner['type']['name'] == 'classic':
                    style_grid[y][x] = 'cyan'
                elif spinner['type']['name'] == 'braille':
                    style_grid[y][x] = 'green'
                elif spinner['type']['name'] == 'circle':
                    style_grid[y][x] = 'blue'
                elif spinner['type']['name'] == 'square':
                    style_grid[y][x] = 'magenta'
                elif spinner['type']['name'] == 'dots':
                    style_grid[y][x] = 'yellow'
                elif spinner['type']['name'] == 'arrows':
                    style_grid[y][x] = 'red'
                else:
                    style_grid[y][x] = 'white'
                
                # Add label
                label = spinner['type']['name']
                if x + len(label) + 2 < self.width:
                    for i, char in enumerate(label):
                        grid[y][x + 2 + i] = char
                        style_grid[y][x + 2 + i] = 'dim white'
        
        # Add title
        self._add_centered_text(grid, style_grid, self.title, self.height - 2, 'bold white')
        
        # Add synchronization indicator
        sync_y = 1
        sync_text = f"Sync: {int(self.phase % len(self.spinner_types[0]['frames']))} / {len(self.spinner_types[0]['frames'])}"
        self._add_centered_text(grid, style_grid, sync_text, sync_y, 'dim white')
        
        return self._grid_to_string(grid, style_grid)


class HackerTerminalEffect(BaseEffect):
    """Hacking simulation with terminal commands."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.terminal_lines = []
        self.current_command = ""
        self.typing_progress = 0
        self.phase = "connecting"  # connecting, scanning, cracking, granted
        self.phase_start_time = 0
        
        # Hacking sequence
        self.commands = [
            ("$ ssh root@mainframe.tldw.ai", "Connecting to mainframe..."),
            ("$ nmap -sS -p- 192.168.1.1", "Scanning ports..."),
            ("$ ./exploit.sh --target=firewall", "Bypassing firewall..."),
            ("$ hashcat -m 0 -a 0 hashes.txt wordlist.txt", "Cracking passwords..."),
            ("$ sudo access --grant-all", "Escalating privileges..."),
        ]
        self.current_command_index = 0
    
    def update(self, elapsed_time):
        """Update hacking simulation."""
        self.phase_start_time += elapsed_time
        
        # Type current command
        if self.current_command_index < len(self.commands):
            cmd, _ = self.commands[self.current_command_index]
            if self.typing_progress < len(cmd):
                self.typing_progress += elapsed_time * 30  # Typing speed
                self.current_command = cmd[:int(self.typing_progress)]
            else:
                # Command complete, add to terminal
                if self.current_command and self.current_command not in [line['text'] for line in self.terminal_lines]:
                    self.terminal_lines.append({
                        'text': self.current_command,
                        'style': 'green'
                    })
                    
                    # Add response
                    _, response = self.commands[self.current_command_index]
                    self.terminal_lines.append({
                        'text': response,
                        'style': 'dim white'
                    })
                    
                    # Progress bar
                    progress = int((self.current_command_index + 1) / len(self.commands) * 20)
                    self.terminal_lines.append({
                        'text': '[' + '█' * progress + '░' * (20 - progress) + '] ' + 
                               f"{(self.current_command_index + 1) * 20}%",
                        'style': 'cyan'
                    })
                    
                    self.current_command_index += 1
                    self.typing_progress = 0
                    self.current_command = ""
        else:
            # All commands complete
            if self.phase != "granted":
                self.phase = "granted"
                self.terminal_lines.append({
                    'text': "",
                    'style': None
                })
                self.terminal_lines.append({
                    'text': "ACCESS GRANTED",
                    'style': 'bold green'
                })
                self.terminal_lines.append({
                    'text': f"Welcome to {self.title}",
                    'style': 'bold white'
                })
        
        # Scroll if too many lines
        while len(self.terminal_lines) > self.height - 4:
            self.terminal_lines.pop(0)
    
    def render(self):
        """Render hacker terminal."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw terminal frame
        for x in range(self.width):
            grid[0][x] = '─'
            grid[self.height - 1][x] = '─'
            style_grid[0][x] = style_grid[self.height - 1][x] = 'green'
        
        for y in range(self.height):
            grid[y][0] = '│'
            grid[y][self.width - 1] = '│'
            style_grid[y][0] = style_grid[y][self.width - 1] = 'green'
        
        # Corners
        grid[0][0] = '┌'
        grid[0][self.width - 1] = '┐'
        grid[self.height - 1][0] = '└'
        grid[self.height - 1][self.width - 1] = '┘'
        
        # Terminal title
        title_text = " TERMINAL v2.0 "
        title_x = 2
        for i, char in enumerate(title_text):
            if title_x + i < self.width - 1:
                grid[0][title_x + i] = char
                style_grid[0][title_x + i] = 'bold green'
        
        # Draw terminal lines
        y_offset = 2
        for line in self.terminal_lines:
            if y_offset < self.height - 2:
                text = line['text']
                style = line['style']
                
                for i, char in enumerate(text):
                    if i + 2 < self.width - 2:
                        grid[y_offset][i + 2] = char
                        if style:
                            style_grid[y_offset][i + 2] = style
                
                y_offset += 1
        
        # Draw current command being typed
        if self.current_command and y_offset < self.height - 2:
            for i, char in enumerate(self.current_command):
                if i + 2 < self.width - 2:
                    grid[y_offset][i + 2] = char
                    style_grid[y_offset][i + 2] = 'green'
            
            # Blinking cursor
            cursor_x = len(self.current_command) + 2
            if cursor_x < self.width - 2 and int(self.phase_start_time * 2) % 2 == 0:
                grid[y_offset][cursor_x] = '█'
                style_grid[y_offset][cursor_x] = 'green'
        
        return self._grid_to_string(grid, style_grid)


#
# End of splash_animations.py
#########################################################################################################################
