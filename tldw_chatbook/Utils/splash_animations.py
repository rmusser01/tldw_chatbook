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