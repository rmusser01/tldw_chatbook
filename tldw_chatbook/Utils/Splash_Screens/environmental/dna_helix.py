"""DNAHelix splash screen effect."""

import math
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("dna_helix")
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