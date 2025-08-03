"""QuantumTunnel splash screen effect."""

from rich.color import Color
import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("quantum_tunnel")
class QuantumTunnelEffect(BaseEffect):
    """3D-like tunnel effect with quantum particles."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "",
        width: int = 80,
        height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        
        self.center_x = width // 2
        self.center_y = height // 2
        self.tunnel_depth = 20
        self.rotation = 0
        
        # Quantum particles
        self.particles = []
        for _ in range(50):
            self.particles.append({
                'z': random.uniform(0, self.tunnel_depth),
                'angle': random.uniform(0, 2 * math.pi),
                'speed': random.uniform(0.1, 0.3),
                'char': random.choice('·•○◌◍◉')
            })
    
    def update(self) -> Optional[str]:
        """Update quantum tunnel effect."""
        elapsed = time.time() - self.start_time
        self.rotation = elapsed * 0.3
        
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw tunnel rings
        for depth in range(1, self.tunnel_depth):
            radius = depth * 2
            z_factor = 1 - (depth / self.tunnel_depth)
            
            # Draw octagonal tunnel shape
            points = 16
            for i in range(points):
                angle = (2 * math.pi * i / points) + self.rotation * z_factor
                
                # 3D projection
                x_3d = radius * math.cos(angle)
                y_3d = radius * math.sin(angle) * 0.5
                
                # Perspective projection
                perspective = 1 / (1 + depth * 0.1)
                x = int(self.center_x + x_3d * perspective)
                y = int(self.center_y + y_3d * perspective)
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    if depth < 5:
                        char = '█'
                    elif depth < 10:
                        char = '▓'
                    elif depth < 15:
                        char = '▒'
                    else:
                        char = '░'
                    
                    grid[y][x] = char
                    
                    # Color based on depth
                    intensity = int(255 * z_factor)
                    style_grid[y][x] = f'rgb({intensity//2},{intensity},{intensity})'
        
        # Update and draw particles
        for particle in self.particles:
            # Move particle forward
            particle['z'] -= particle['speed']
            if particle['z'] <= 0:
                particle['z'] = self.tunnel_depth
                particle['angle'] = random.uniform(0, 2 * math.pi)
            
            # Calculate position
            radius = particle['z'] * 1.5
            x_3d = radius * math.cos(particle['angle'] + self.rotation)
            y_3d = radius * math.sin(particle['angle'] + self.rotation) * 0.5
            
            perspective = 1 / (1 + particle['z'] * 0.1)
            x = int(self.center_x + x_3d * perspective)
            y = int(self.center_y + y_3d * perspective)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = particle['char']
                intensity = int(255 * (1 - particle['z'] / self.tunnel_depth))
                style_grid[y][x] = f'rgb(0,{intensity},{intensity//2})'
        
        # Title emerges from tunnel
        if elapsed > 2.0:
            title_progress = min(1.0, (elapsed - 2.0) / 1.0)
            title_size = 1.0 + (1 - title_progress) * 2  # Shrink from large to normal
            
            # Draw title with perspective effect
            for i, char in enumerate(self.title):
                x = int(self.center_x - len(self.title) * title_size / 2 + i * title_size)
                y = self.center_y
                
                if 0 <= x < self.width:
                    grid[y][x] = char
                    style_grid[y][x] = f'bold rgb(0,{int(255 * title_progress)},255)'
        
        # Subtitle
        if self.subtitle and elapsed > 3.0:
            subtitle_x = self.center_x - len(self.subtitle) // 2
            subtitle_y = self.center_y + 2
            for i, char in enumerate(self.subtitle):
                if 0 <= subtitle_x + i < self.width:
                    grid[subtitle_y][subtitle_x + i] = char
                    style_grid[subtitle_y][subtitle_x + i] = 'cyan'
        
        # Convert to string
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                char = grid[y][x]
                style = style_grid[y][x]
                if style:
                    line += f"[{style}]{char}[/{style.split()[0]}]"
                else:
                    line += char
            lines.append(line)
        return '\n'.join(lines)