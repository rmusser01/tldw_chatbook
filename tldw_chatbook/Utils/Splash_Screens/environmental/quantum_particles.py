"""QuantumParticles splash screen effect."""
import time

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("quantum_particles")
class QuantumParticlesEffect(BaseEffect):
    """Quantum particles with superposition and entanglement effects."""
    
    def __init__(self, parent, title="TLDW Chatbook", subtitle="", width=80, height=24, speed=0.05, **kwargs):
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
    
    def update(self) -> Optional[str]:
        """Update quantum particles."""
        elapsed_time = time.time() - self.start_time
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
        
        # Return the rendered content
        return self.render()
    
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