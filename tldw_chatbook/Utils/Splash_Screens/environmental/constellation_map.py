"""ConstellationMap splash screen effect."""
import time

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("constellation_map")
class ConstellationMapEffect(BaseEffect):
    """Stars forming constellations that spell out the logo."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.1, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.stars = []
        self.connections = []
        self.shooting_stars = []
        self.reveal_progress = 0
        
        # Create background stars
        for _ in range(50):
            self.stars.append({
                'x': random.randint(0, width - 1),
                'y': random.randint(0, height - 1),
                'brightness': random.choice(['·', '•', '*']),
                'twinkle': random.random(),
                'constellation': False
            })
        
        # Create constellation pattern (simplified TLDW shape)
        self._create_constellation()
    
    def _create_constellation(self):
        """Create constellation in shape of letters."""
        # T constellation
        cx = self.width // 4
        cy = self.height // 2
        t_stars = [
            (cx - 4, cy - 4), (cx, cy - 4), (cx + 4, cy - 4),  # Top of T
            (cx, cy - 2), (cx, cy), (cx, cy + 2), (cx, cy + 4)  # Stem of T
        ]
        
        # L constellation
        cx = self.width // 2
        l_stars = [
            (cx - 4, cy - 4), (cx - 4, cy - 2), (cx - 4, cy), (cx - 4, cy + 2), (cx - 4, cy + 4),  # Vertical
            (cx - 2, cy + 4), (cx, cy + 4), (cx + 2, cy + 4)  # Horizontal
        ]
        
        # Add constellation stars
        for x, y in t_stars + l_stars:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.stars.append({
                    'x': x,
                    'y': y,
                    'brightness': '★',
                    'twinkle': 0,
                    'constellation': True,
                    'revealed': False
                })
        
        # Create connections for T
        for i in range(3):  # Top horizontal line
            self.connections.append((t_stars[i], t_stars[i + 1] if i < 2 else t_stars[1]))
        for i in range(len(t_stars) - 4):  # Vertical line
            if i + 4 < len(t_stars):
                self.connections.append((t_stars[i + 3], t_stars[i + 4]))
        
        # Create connections for L
        for i in range(4):  # Vertical line
            if i + 1 < 5:
                self.connections.append((l_stars[i], l_stars[i + 1]))
        for i in range(5, 7):  # Horizontal line
            if i + 1 < len(l_stars):
                self.connections.append((l_stars[i], l_stars[i + 1]))
    
    def update(self) -> Optional[str]:
        """Update constellation animation."""
        elapsed_time = time.time() - self.start_time
        # Twinkle stars
        for star in self.stars:
            if not star['constellation']:
                star['twinkle'] = (star['twinkle'] + elapsed_time * random.uniform(1, 3)) % 1.0
        
        # Reveal constellation gradually
        self.reveal_progress += elapsed_time * 0.5
        revealed_count = int(self.reveal_progress * len([s for s in self.stars if s['constellation']]))
        
        constellation_stars = [s for s in self.stars if s['constellation']]
        for i, star in enumerate(constellation_stars):
            if i < revealed_count:
                star['revealed'] = True
        
        # Create shooting stars
        if random.random() < 0.02:
            self.shooting_stars.append({
                'x': random.randint(0, self.width),
                'y': 0,
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(0.5, 1.5),
                'trail': []
            })
        
        # Update shooting stars
        for star in self.shooting_stars[:]:
            star['x'] += star['vx']
            star['y'] += star['vy']
            star['trail'].append((int(star['x']), int(star['y'])))
            if len(star['trail']) > 5:
                star['trail'].pop(0)
            
            if star['y'] >= self.height:
                self.shooting_stars.remove(star)
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render constellation map."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw background stars
        for star in self.stars:
            x, y = int(star['x']), int(star['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                if star['constellation'] and star.get('revealed', False):
                    grid[y][x] = star['brightness']
                    style_grid[y][x] = 'bold yellow'
                elif not star['constellation']:
                    if star['twinkle'] > 0.7:
                        grid[y][x] = star['brightness']
                        style_grid[y][x] = 'white'
                    else:
                        grid[y][x] = '·'
                        style_grid[y][x] = 'dim white'
        
        # Draw constellation connections
        if self.reveal_progress > 0.3:
            for (x1, y1), (x2, y2) in self.connections:
                star1 = next((s for s in self.stars if s['x'] == x1 and s['y'] == y1 and s.get('revealed')), None)
                star2 = next((s for s in self.stars if s['x'] == x2 and s['y'] == y2 and s.get('revealed')), None)
                if star1 and star2:
                    self._draw_constellation_line(grid, style_grid, x1, y1, x2, y2)
        
        # Draw shooting stars
        for star in self.shooting_stars:
            for i, (x, y) in enumerate(star['trail']):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if i == len(star['trail']) - 1:
                        grid[y][x] = '✦'
                        style_grid[y][x] = 'bold white'
                    else:
                        grid[y][x] = '·'
                        style_grid[y][x] = 'dim white'
        
        # Add title when constellation is mostly revealed
        if self.reveal_progress > 0.7:
            self._add_centered_text(grid, style_grid, self.title, self.height - 3, 'bold white')
        
        return self._grid_to_string(grid, style_grid)
    
    def _draw_constellation_line(self, grid, style_grid, x1, y1, x2, y2):
        """Draw a faint line between constellation stars."""
        steps = max(abs(x2 - x1), abs(y2 - y1)) * 2
        if steps == 0:
            return
        
        for i in range(1, steps):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                if grid[y][x] == ' ':
                    grid[y][x] = '·'
                    style_grid[y][x] = 'dim yellow'