"""CircuitTrace splash screen effect."""
import time

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("circuit_trace")
class CircuitTraceEffect(BaseEffect):
    """PCB circuit traces being drawn with electrical signals."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.02, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.traces = []
        self.components = []
        self.signals = []
        self.trace_progress = 0
        
        # Generate circuit layout
        self._generate_circuit()
    
    def _generate_circuit(self):
        """Generate a circuit board layout."""
        # Main bus lines
        for y in [5, 10, 15, 20]:
            self.traces.append({
                'start': (5, y),
                'end': (self.width - 5, y),
                'drawn': 0,
                'type': 'horizontal'
            })
        
        # Vertical connections
        for x in range(10, self.width - 10, 15):
            y_start = random.choice([5, 10])
            y_end = random.choice([15, 20])
            self.traces.append({
                'start': (x, y_start),
                'end': (x, y_end),
                'drawn': 0,
                'type': 'vertical'
            })
        
        # Add components
        component_types = [
            {'symbol': '[R]', 'name': 'resistor'},
            {'symbol': '[C]', 'name': 'capacitor'},
            {'symbol': '[D]', 'name': 'diode'},
            {'symbol': '[U]', 'name': 'chip'}
        ]
        
        for _ in range(10):
            self.components.append({
                'x': random.randint(10, self.width - 10),
                'y': random.choice([5, 10, 15, 20]),
                'type': random.choice(component_types),
                'placed': False
            })
    
    def update(self) -> Optional[str]:
        """Update circuit trace animation."""
        elapsed_time = time.time() - self.start_time
        self.trace_progress = min(1.0, self.trace_progress + elapsed_time * 0.5)
        
        # Update trace drawing
        for trace in self.traces:
            trace['drawn'] = min(1.0, trace['drawn'] + elapsed_time * 2)
        
        # Create electrical signals
        if random.random() < 0.1 and self.trace_progress > 0.3:
            trace = random.choice([t for t in self.traces if t['drawn'] > 0.5])
            self.signals.append({
                'trace': trace,
                'position': 0,
                'speed': random.uniform(20, 40)
            })
        
        # Update signals
        for signal in self.signals[:]:
            signal['position'] += signal['speed'] * elapsed_time
            if signal['position'] > 1.0:
                self.signals.remove(signal)
        
        # Place components
        for comp in self.components:
            if not comp['placed'] and self.trace_progress > random.uniform(0.2, 0.8):
                comp['placed'] = True
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render circuit board."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw PCB background
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < 0.02:
                    grid[y][x] = '·'
                    style_grid[y][x] = 'dim green'
        
        # Draw traces
        for trace in self.traces:
            x1, y1 = trace['start']
            x2, y2 = trace['end']
            length = max(abs(x2 - x1), abs(y2 - y1))
            drawn_length = int(length * trace['drawn'])
            
            if trace['type'] == 'horizontal':
                for i in range(drawn_length + 1):
                    x = x1 + i if x2 > x1 else x1 - i
                    if 0 <= x < self.width:
                        grid[y1][x] = '═'
                        style_grid[y1][x] = 'yellow'
            else:  # vertical
                for i in range(drawn_length + 1):
                    y = y1 + i if y2 > y1 else y1 - i
                    if 0 <= y < self.height:
                        grid[y][x1] = '║'
                        style_grid[y][x1] = 'yellow'
        
        # Draw components
        for comp in self.components:
            if comp['placed']:
                x, y = comp['x'], comp['y']
                symbol = comp['type']['symbol']
                if x + len(symbol) < self.width and 0 <= y < self.height:
                    for i, char in enumerate(symbol):
                        grid[y][x + i] = char
                        style_grid[y][x + i] = 'bold cyan'
        
        # Draw signals
        for signal in self.signals:
            trace = signal['trace']
            x1, y1 = trace['start']
            x2, y2 = trace['end']
            
            # Calculate signal position
            t = signal['position']
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[y][x] = '●'
                style_grid[y][x] = 'bold white'
        
        # Add title when circuit is mostly complete
        if self.trace_progress > 0.7:
            self._add_centered_text(grid, style_grid, self.title, self.height // 2, 'bold white')
        
        return self._grid_to_string(grid, style_grid)