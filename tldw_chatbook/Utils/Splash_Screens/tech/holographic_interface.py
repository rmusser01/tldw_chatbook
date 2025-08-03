"""HolographicInterface splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("holographic_interface")
class HolographicInterfaceEffect(BaseEffect):
    """Simulates a holographic UI startup sequence."""
    
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
        
        # HUD elements
        self.panels = []
        self.scan_line = 0
        self.boot_progress = 0
        self.flicker_intensity = 1.0
        
        # Initialize panels
        self._init_panels()
    
    def _init_panels(self):
        """Initialize holographic panels."""
        # Main display panel
        self.panels.append({
            'x': self.width // 2 - 20,
            'y': self.height // 2 - 5,
            'w': 40,
            'h': 10,
            'type': 'main',
            'active': False,
            'progress': 0
        })
        
        # Side panels
        self.panels.append({
            'x': 2,
            'y': 2,
            'w': 20,
            'h': 8,
            'type': 'stats',
            'active': False,
            'progress': 0
        })
        
        self.panels.append({
            'x': self.width - 22,
            'y': 2,
            'w': 20,
            'h': 8,
            'type': 'info',
            'active': False,
            'progress': 0
        })
    
    def update(self) -> Optional[str]:
        """Update holographic interface."""
        elapsed = time.time() - self.start_time
        self.boot_progress = min(1.0, elapsed / 3.0)
        
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Holographic flicker
        self.flicker_intensity = 0.7 + 0.3 * math.sin(elapsed * 20)
        
        # Scanning effect
        self.scan_line = int(elapsed * 10) % self.height
        for x in range(self.width):
            if random.random() < 0.3:
                grid[self.scan_line][x] = '─'
                style_grid[self.scan_line][x] = f'rgb(0,{int(255 * self.flicker_intensity)},255)'
        
        # Activate panels progressively
        for i, panel in enumerate(self.panels):
            activation_time = i * 0.5
            if elapsed > activation_time:
                panel['active'] = True
                panel['progress'] = min(1.0, (elapsed - activation_time) / 0.5)
        
        # Draw panels
        for panel in self.panels:
            if panel['active']:
                self._draw_panel(grid, style_grid, panel)
        
        # Draw main title in center panel after boot
        if elapsed > 2.0:
            main_panel = self.panels[0]
            title_x = main_panel['x'] + (main_panel['w'] - len(self.title)) // 2
            title_y = main_panel['y'] + main_panel['h'] // 2
            
            for i, char in enumerate(self.title):
                if title_x + i < main_panel['x'] + main_panel['w']:
                    grid[title_y][title_x + i] = char
                    style_grid[title_y][title_x + i] = 'bold cyan'
            
            # Subtitle
            if self.subtitle:
                subtitle_x = main_panel['x'] + (main_panel['w'] - len(self.subtitle)) // 2
                subtitle_y = title_y + 2
                for i, char in enumerate(self.subtitle):
                    if subtitle_x + i < main_panel['x'] + main_panel['w']:
                        grid[subtitle_y][subtitle_x + i] = char
                        style_grid[subtitle_y][subtitle_x + i] = 'rgb(0,200,200)'
        
        # Add holographic artifacts
        for _ in range(30):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if grid[y][x] == ' ' and random.random() < 0.1:
                grid[y][x] = '·'
                style_grid[y][x] = f'rgb(0,{int(100 * self.flicker_intensity)},100)'
        
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
    
    def _draw_panel(self, grid, style_grid, panel):
        """Draw a holographic panel."""
        x, y, w, h = panel['x'], panel['y'], panel['w'], panel['h']
        progress = panel['progress']
        
        # Draw panel frame with progress
        frame_chars = {
            'top_left': '┌', 'top_right': '┐',
            'bottom_left': '└', 'bottom_right': '┘',
            'horizontal': '─', 'vertical': '│'
        }
        
        # Top and bottom borders
        for i in range(int(w * progress)):
            if x + i < self.width:
                grid[y][x + i] = frame_chars['horizontal']
                grid[y + h - 1][x + i] = frame_chars['horizontal']
                style_grid[y][x + i] = style_grid[y + h - 1][x + i] = 'cyan'
        
        # Side borders
        for i in range(int(h * progress)):
            if y + i < self.height:
                grid[y + i][x] = frame_chars['vertical']
                grid[y + i][x + w - 1] = frame_chars['vertical']
                style_grid[y + i][x] = style_grid[y + i][x + w - 1] = 'cyan'
        
        # Corners
        if progress > 0.5:
            grid[y][x] = frame_chars['top_left']
            grid[y][x + w - 1] = frame_chars['top_right']
            grid[y + h - 1][x] = frame_chars['bottom_left']
            grid[y + h - 1][x + w - 1] = frame_chars['bottom_right']
            
            for pos in [(y, x), (y, x + w - 1), (y + h - 1, x), (y + h - 1, x + w - 1)]:
                style_grid[pos[0]][pos[1]] = 'bold cyan'
        
        # Panel content based on type
        if progress > 0.8:
            if panel['type'] == 'stats':
                stats = ['CPU: 87%', 'MEM: 4.2GB', 'NET: OK']
                for i, stat in enumerate(stats):
                    stat_y = y + 2 + i
                    if stat_y < y + h - 1:
                        for j, char in enumerate(stat):
                            if x + 2 + j < x + w - 1:
                                grid[stat_y][x + 2 + j] = char
                                style_grid[stat_y][x + 2 + j] = 'green'
            
            elif panel['type'] == 'info':
                info = ['v2.0.1', 'READY', '████████']
                for i, text in enumerate(info):
                    text_y = y + 2 + i
                    if text_y < y + h - 1:
                        for j, char in enumerate(text):
                            if x + 2 + j < x + w - 1:
                                grid[text_y][x + 2 + j] = char
                                style_grid[text_y][x + 2 + j] = 'yellow'