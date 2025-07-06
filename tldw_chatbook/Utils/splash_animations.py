# splash_animations.py
# Animation effects library for splash screens
# Provides various animation effects like Matrix rain, glitch, typewriter, etc.

import random
import time
import math
from typing import List, Optional, Tuple, Any
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
        title: str = "TLDW chatbook",
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
        title: str = "TLDW Chatbook",
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
        title: str = "TLDW Chatbook",
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