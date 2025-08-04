"""CustomImage splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("custom_image")
class CustomImageEffect(BaseEffect):
    """Display a custom image as ASCII art."""
    
    def __init__(
        self,
        parent_widget: Any,
        image_path: str,
        width: int = 80,
        height: int = 24,
        speed: float = 0.1,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.width = width
        self.height = height
        self.speed = speed
        self.image_path = image_path
        self.ascii_art = None
        self.error_message = None
        self.fade_progress = 0.0
        
        # ASCII characters for different brightness levels
        self.ascii_chars = " .:-=+*#%@"
        
        # Try to load and convert the image
        self._load_and_convert_image()
    
    def _load_and_convert_image(self):
        """Load image and convert to ASCII art."""
        try:
            # Check if PIL is available
            try:
                from PIL import Image
            except ImportError:
                self.error_message = "PIL/Pillow not installed. Install with: pip install pillow"
                return
            
            # Load the image
            try:
                img = Image.open(self.image_path)
            except Exception as e:
                self.error_message = f"Could not load image: {str(e)}"
                return
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Calculate aspect ratio correction (terminal chars are ~2x taller than wide)
            aspect_ratio = img.width / img.height
            
            # Reserve space for title
            available_height = self.height - 6
            available_width = self.width - 4
            
            # Calculate new dimensions
            if aspect_ratio > available_width / (available_height * 2):
                # Image is wider
                new_width = available_width
                new_height = int(new_width / aspect_ratio / 2)
            else:
                # Image is taller
                new_height = available_height
                new_width = int(new_height * aspect_ratio * 2)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to ASCII
            ascii_lines = []
            for y in range(new_height):
                line = ""
                for x in range(new_width):
                    brightness = img.getpixel((x, y))
                    char_index = int(brightness / 256 * len(self.ascii_chars))
                    char_index = min(char_index, len(self.ascii_chars) - 1)
                    line += self.ascii_chars[char_index]
                ascii_lines.append(line)
            
            self.ascii_art = ascii_lines
            
        except Exception as e:
            self.error_message = f"Error converting image: {str(e)}"
    
    def update(self) -> Optional[str]:
        """Update the custom image display."""
        elapsed = time.time() - self.start_time
        
        # Fade in effect
        self.fade_progress = min(1.0, elapsed / 1.5)
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Title
        title = "TLDW CHATBOOK"
        subtitle = "Custom Splash Screen"
        
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[1][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[3][subtitle_x + i] = char
        
        # Display content
        if self.error_message:
            # Show error message
            error_lines = self.error_message.split('\n')
            start_y = (self.height - len(error_lines)) // 2
            for i, line in enumerate(error_lines):
                start_x = (self.width - len(line)) // 2
                for j, char in enumerate(line):
                    if 0 <= start_x + j < self.width and 0 <= start_y + i < self.height:
                        grid[start_y + i][start_x + j] = char
        elif self.ascii_art:
            # Display ASCII art with fade-in
            start_y = 5
            start_x = (self.width - len(self.ascii_art[0])) // 2
            
            for i, line in enumerate(self.ascii_art):
                if start_y + i >= self.height - 2:
                    break
                for j, char in enumerate(line):
                    if 0 <= start_x + j < self.width:
                        # Apply fade-in effect
                        if random.random() < self.fade_progress:
                            grid[start_y + i][start_x + j] = char
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 1 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 3 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif self.error_message and y >= (self.height - 5) // 2 and y <= (self.height + 5) // 2:
                    line += f"[bold red]{char}[/bold red]"
                else:
                    # Apply brightness-based coloring for ASCII art
                    if char in self.ascii_chars and char != ' ':
                        brightness = self.ascii_chars.index(char) / len(self.ascii_chars)
                        if brightness < 0.3:
                            line += f"[dim white]{char}[/dim white]"
                        elif brightness < 0.6:
                            line += f"[white]{char}[/white]"
                        else:
                            line += f"[bold white]{char}[/bold white]"
                    else:
                        line += char
            lines.append(line)
        
        return '\n'.join(lines)