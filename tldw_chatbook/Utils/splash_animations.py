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
                escaped_char = char.replace('[', r'\[').replace(']', r'\]')
                
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
                escaped_char = char.replace('[', r'\[')
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
            output_lines.append(f"[{self.text_style}]{' ' * pad_above}{self.text_above.replace('[', r'\[')}{' ' * pad_above}[/{self.text_style}]")
        else:
            output_lines.append("") # Keep spacing consistent

        pad_bar = (self.width - len(self.bar_frame_content)) // 2
        output_lines.append(f"{' ' * pad_bar}{styled_bar}")

        if text_below_formatted:
            pad_below = (self.width - len(text_below_formatted)) // 2
            output_lines.append(f"[{self.text_style}]{' ' * pad_below}{text_below_formatted.replace('[', r'\[')}{' ' * pad_below}[/{self.text_style}]")
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
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
                            title_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
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
                        escaped_char = char.replace('[', r'\[')

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
                    line_segments.append(f"[{style}]{char.replace('[',r'\[')}[/{style}]")
                else: # Space, apply default path style or background
                    line_segments.append(f"[{self.path_style}] {[/ {self.path_style}]}")
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
                line_segments.append(f"[{style}]{char.replace('[',r'\[')}[/{style}]")
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
                        title_line_segments.append(f"[{style}]{char.replace('[',r'\[')}[/{style}]")
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
                    line_segments.append(f"[{style}]{char.replace('[',r'\[')}[/{style}]")
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
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
                            title_line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
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
                current_line_content = f"[{self.title_style}]{current_line_content.replace('[', r'\[')}[/{self.title_style}]"
            elif r_idx == subtitle_row:
                padding = (self.width - len(self.subtitle)) // 2
                current_line_content = f"{' ' * padding}{self.subtitle}{' ' * (self.width - len(self.subtitle) - padding)}"
                current_line_content = f"[{self.subtitle_style}]{current_line_content.replace('[', r'\[')}[/{self.subtitle_style}]"
            elif code_block_start_row <= r_idx < code_block_end_row:
                code_line_index = r_idx - code_block_start_row
                code_text = self.code_lines[code_line_index]
                # Ensure code_text is padded to full width if needed, or handled by terminal
                current_line_content = f"[{self.code_line_style}]{code_text.replace('[', r'\[')}{' ' * (self.width - len(code_text))}[/{self.code_line_style}]"
            else:
                current_line_content = ' ' * self.width

            output_lines.append(current_line_content)

        return '\n'.join(output_lines)