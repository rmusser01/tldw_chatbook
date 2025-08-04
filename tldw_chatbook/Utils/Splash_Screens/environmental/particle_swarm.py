"""ParticleSwarm splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect


@register_effect("particle_swarm")
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