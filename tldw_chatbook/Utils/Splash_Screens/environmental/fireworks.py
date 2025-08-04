"""Fireworks splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect


@register_effect("fireworks")
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