"""
Example custom animation effect for TLDW CLI splash screens.
Place this file in your project and import it in splash_animations.py
"""

import math
import time
from typing import Optional, Any
from splash_animations import BaseEffect


class FireEffect(BaseEffect):
    """Creates an animated ASCII fire effect."""
    
    def __init__(
        self, 
        parent_widget: Any, 
        content: str,
        fire_height: int = 5,
        fire_chars: str = " .':^*#",
        speed: float = 0.1,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.fire_height = fire_height
        self.fire_chars = fire_chars
        self.speed = speed
        
        # Split content into lines for processing
        self.content_lines = content.strip().split('\n')
        self.width = max(len(line) for line in self.content_lines)
        
        # Initialize fire grid
        self.fire_grid = [[0 for _ in range(self.width)] for _ in range(fire_height)]
        
    def update(self) -> Optional[str]:
        """Update fire animation and return the frame."""
        # Update fire simulation
        self._update_fire()
        
        # Build output
        output_lines = []
        
        # Add fire effect at bottom
        for row in reversed(self.fire_grid):
            fire_line = ""
            for intensity in row:
                char_index = min(int(intensity * len(self.fire_chars)), len(self.fire_chars) - 1)
                fire_line += self.fire_chars[char_index]
            output_lines.append(fire_line)
        
        # Add content above fire
        output_lines.extend(self.content_lines)
        
        return '\n'.join(reversed(output_lines))
    
    def _update_fire(self):
        """Update the fire simulation grid."""
        # Random heat at bottom row
        for x in range(self.width):
            if x % 2 == self.frame_count % 2:  # Alternating pattern
                self.fire_grid[0][x] = 0.8 + (0.2 * math.sin(time.time() * 3 + x))
            else:
                self.fire_grid[0][x] *= 0.9
        
        # Propagate heat upward with turbulence
        for y in range(1, self.fire_height):
            for x in range(self.width):
                # Average heat from below with some randomness
                heat = 0
                count = 0
                
                for dx in [-1, 0, 1]:
                    nx = x + dx
                    if 0 <= nx < self.width:
                        heat += self.fire_grid[y-1][nx]
                        count += 1
                
                if count > 0:
                    # Add turbulence based on position and time
                    turbulence = 0.1 * math.sin(time.time() * 2 + x * 0.5)
                    self.fire_grid[y][x] = (heat / count) * 0.95 + turbulence
                    self.fire_grid[y][x] = max(0, min(1, self.fire_grid[y][x]))


class RainbowWaveEffect(BaseEffect):
    """Creates a rainbow wave effect across the text."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        wave_speed: float = 2.0,
        wave_height: float = 3.0,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.wave_speed = wave_speed
        self.wave_height = wave_height
        self.lines = content.split('\n')
        
        # Rainbow colors in RGB
        self.colors = [
            (255, 0, 0),     # Red
            (255, 127, 0),   # Orange
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (75, 0, 130),    # Indigo
            (148, 0, 211),   # Violet
        ]
    
    def update(self) -> Optional[str]:
        """Create rainbow wave effect."""
        from rich.text import Text
        from rich.style import Style
        
        output = Text()
        time_offset = time.time() - self.start_time
        
        for y, line in enumerate(self.lines):
            # Calculate wave offset for this line
            wave_offset = math.sin(y * 0.3 + time_offset * self.wave_speed) * self.wave_height
            line_with_offset = " " * int(max(0, wave_offset)) + line
            
            # Apply rainbow colors
            for x, char in enumerate(line_with_offset):
                if char != ' ':
                    # Calculate color based on position and time
                    color_index = (x + y + int(time_offset * 10)) % len(self.colors)
                    color = self.colors[color_index]
                    style = Style(color=f"rgb({color[0]},{color[1]},{color[2]})")
                    output.append(char, style=style)
                else:
                    output.append(char)
            
            output.append('\n')
        
        return output


class ParticleEffect(BaseEffect):
    """Creates a particle system effect around the content."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        num_particles: int = 20,
        particle_chars: str = "✦✧✨⋆∘°",
        boundary_padding: int = 5,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content = content
        self.num_particles = num_particles
        self.particle_chars = particle_chars
        self.boundary_padding = boundary_padding
        
        # Parse content dimensions
        self.lines = content.split('\n')
        self.content_height = len(self.lines)
        self.content_width = max(len(line) for line in self.lines) if self.lines else 0
        
        # Initialize particles
        self.particles = []
        for _ in range(num_particles):
            self.particles.append({
                'x': random.uniform(-boundary_padding, self.content_width + boundary_padding),
                'y': random.uniform(-boundary_padding, self.content_height + boundary_padding),
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(-0.5, 0.5),
                'char': random.choice(self.particle_chars),
                'life': random.uniform(0, 1),
            })
    
    def update(self) -> Optional[str]:
        """Update particle positions and render."""
        import random
        
        # Update particles
        for particle in self.particles:
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Update life
            particle['life'] -= 0.02
            
            # Respawn dead particles
            if particle['life'] <= 0:
                particle['x'] = random.uniform(-self.boundary_padding, self.content_width + self.boundary_padding)
                particle['y'] = random.choice([-self.boundary_padding, self.content_height + self.boundary_padding])
                particle['vx'] = random.uniform(-0.5, 0.5)
                particle['vy'] = random.uniform(-0.5, 0.5)
                particle['char'] = random.choice(self.particle_chars)
                particle['life'] = 1.0
        
        # Create display grid
        total_width = self.content_width + 2 * self.boundary_padding
        total_height = self.content_height + 2 * self.boundary_padding
        display = [[' ' for _ in range(total_width)] for _ in range(total_height)]
        
        # Place particles
        for particle in self.particles:
            x = int(particle['x'] + self.boundary_padding)
            y = int(particle['y'] + self.boundary_padding)
            if 0 <= x < total_width and 0 <= y < total_height:
                if particle['life'] > 0.5:
                    display[y][x] = particle['char']
                elif particle['life'] > 0.25:
                    display[y][x] = '·'
                else:
                    display[y][x] = '.'
        
        # Overlay content in center
        for i, line in enumerate(self.lines):
            y = i + self.boundary_padding
            for j, char in enumerate(line):
                x = j + self.boundary_padding
                if char != ' ':
                    display[y][x] = char
        
        # Convert to string
        return '\n'.join(''.join(row) for row in display)


# Example usage in splash_screen.py:
"""
# Import the custom effect
from examples.custom_splash_cards.custom_animation_effect import FireEffect

# Add to built_in_cards
"fire_card": {
    "type": "animated",
    "effect": "fire",  # This will map to FireEffect
    "content": get_ascii_art("default"),
    "style": "bold red on black",
    "fire_height": 5,
    "animation_speed": 0.1
}

# In _start_card_animation method, add:
elif effect_type == "fire":
    self.effect_handler = FireEffect(
        self,
        content=self.card_data.get("content", self.DEFAULT_SPLASH),
        fire_height=self.card_data.get("fire_height", 5),
        speed=self.card_data.get("animation_speed", 0.1)
    )
"""