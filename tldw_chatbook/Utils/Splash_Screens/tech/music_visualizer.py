"""MusicVisualizer splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("music_visualizer")
class MusicVisualizerEffect(BaseEffect):
    """Animated music visualizer with notes and instruments."""
    
    def __init__(
        self,
        parent_widget: Any,
        width: int = 80,
        height: int = 24,
        speed: float = 0.1,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.width = width
        self.height = height
        self.speed = speed
        
        # Musical elements
        self.notes = ['♪', '♫', '♩', '♬']
        self.piano_keys = "█▄█▄█ █▄█▄█▄█ █▄█▄█ █▄█▄█▄█"
        
        # Waveform data
        self.waveform = []
        for i in range(self.width):
            self.waveform.append(random.uniform(-1, 1))
        
        # Floating notes
        self.floating_notes = []
        for _ in range(15):
            self.floating_notes.append({
                'x': random.randint(0, self.width),
                'y': random.randint(self.height // 2, self.height),
                'note': random.choice(self.notes),
                'speed': random.uniform(0.5, 1.5),
                'drift': random.uniform(-0.3, 0.3)
            })
        
        # Beat counter
        self.beat = 0
        
    def update(self) -> Optional[str]:
        """Update the music visualizer animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Update beat
        self.beat = int(elapsed * 4) % 4
        
        # Draw title with pulsing effect
        title = "TLDW CHATBOOK"
        subtitle = "♪ Let the Music Play ♪"
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        # Pulse title on beat
        if self.beat == 0:
            title = f"[{title}]"
            title_x -= 1
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[4][subtitle_x + i] = char
        
        # Draw piano keys
        piano_y = self.height - 8
        piano_x = (self.width - len(self.piano_keys)) // 2
        for i, char in enumerate(self.piano_keys):
            if 0 <= piano_x + i < self.width:
                grid[piano_y][piano_x + i] = char
                if char == '▄':
                    grid[piano_y - 1][piano_x + i] = '█'
        
        # Highlight playing keys
        if self.beat == 0:
            highlight_keys = [5, 12, 19]
        elif self.beat == 1:
            highlight_keys = [7, 14, 21]
        elif self.beat == 2:
            highlight_keys = [9, 16, 23]
        else:
            highlight_keys = [3, 10, 17]
        
        for key in highlight_keys:
            if 0 <= piano_x + key < self.width:
                grid[piano_y + 1][piano_x + key] = '▀'
        
        # Draw waveform
        waveform_y = 12
        amplitude = 3
        for x in range(self.width):
            # Update waveform
            self.waveform[x] = math.sin(elapsed * 5 + x * 0.1) * math.sin(elapsed * 2)
            
            # Draw waveform
            y_offset = int(self.waveform[x] * amplitude)
            y = waveform_y + y_offset
            if 0 <= y < self.height:
                if abs(y_offset) > 1:
                    grid[y][x] = '═'
                else:
                    grid[y][x] = '─'
        
        # Update and draw floating notes
        for note in self.floating_notes:
            # Update position
            note['y'] -= note['speed'] * self.speed
            note['x'] += note['drift']
            
            # Reset if off screen
            if note['y'] < 6:
                note['y'] = random.randint(self.height - 10, self.height - 2)
                note['x'] = random.randint(10, self.width - 10)
                note['note'] = random.choice(self.notes)
            
            # Draw note
            x = int(note['x'])
            y = int(note['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = note['note']
        
        # Draw staff lines
        staff_y = 8
        for y in range(staff_y, staff_y + 5):
            if y < self.height:
                for x in range(10, self.width - 10):
                    if grid[y][x] == ' ':
                        grid[y][x] = '─'
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Title
                    if char in '[]':
                        line += f"[bold yellow]{char}[/bold yellow]"
                    else:
                        line += f"[bold white]{char}[/bold white]"
                elif y == 4 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif char in '♪♫♩♬':  # Notes
                    colors = ['cyan', 'magenta', 'yellow', 'green']
                    color = colors[self.notes.index(char)]
                    line += f"[bold {color}]{char}[/bold {color}]"
                elif char in '█▄':  # Piano keys
                    if grid[y + 1][x] == '▀':  # Highlighted key
                        line += f"[bold yellow]{char}[/bold yellow]"
                    else:
                        line += f"[white]{char}[/white]"
                elif char == '▀':  # Key highlight
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char in '─═':  # Waveform and staff
                    if 10 <= y <= 14:  # Waveform area
                        line += f"[green]{char}[/green]"
                    else:
                        line += f"[dim white]{char}[/dim white]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)