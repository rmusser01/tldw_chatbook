"""CircuitBoard splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("circuit_board")
class CircuitBoardEffect(BaseEffect):
    """Animated circuit board traces that light up and connect different components."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Connecting systems...",
        width: int = 80,
        height: int = 24,
        trace_speed: float = 20.0,  # Characters per second
        trace_chars: str = "─│┌┐└┘├┤┬┴┼",
        node_chars: str = "◆◇○●□■",
        active_color: str = "bright_green",
        inactive_color: str = "dim green",
        node_color: str = "yellow",
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.trace_speed = trace_speed
        self.trace_chars = trace_chars
        self.node_chars = node_chars
        self.active_color = active_color
        self.inactive_color = inactive_color
        self.node_color = node_color
        self.title_style = title_style
        
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
        self.active_cells = set()
        self.nodes = []
        self.traces = []
        self.current_trace_progress = 0.0
        self.last_update_time = time.time()
        
        self._generate_circuit()
        
    def _generate_circuit(self):
        """Generate a random circuit board layout."""
        # Place nodes
        num_nodes = random.randint(6, 10)
        for _ in range(num_nodes):
            x = random.randint(5, self.display_width - 5)
            y = random.randint(2, self.display_height - 3)
            node_char = random.choice(self.node_chars)
            self.nodes.append((x, y, node_char))
            self.grid[y][x] = node_char
        
        # Connect nodes with traces
        for i in range(len(self.nodes) - 1):
            start = self.nodes[i]
            end = self.nodes[i + 1]
            path = self._create_path(start[:2], end[:2])
            if path:
                self.traces.append(path)
    
    def _create_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a path between two points using Manhattan routing."""
        x1, y1 = start
        x2, y2 = end
        path = []
        
        # Simple L-shaped path
        if random.random() < 0.5:
            # Horizontal first
            for x in range(min(x1, x2), max(x1, x2) + 1):
                path.append((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                path.append((x2, y))
        else:
            # Vertical first
            for y in range(min(y1, y2), max(y1, y2) + 1):
                path.append((x1, y))
            for x in range(min(x1, x2), max(x1, x2) + 1):
                path.append((x, y2))
        
        return path
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update trace animation
        self.current_trace_progress += self.trace_speed * delta_time
        
        # Calculate which cells should be active
        total_cells = sum(len(trace) for trace in self.traces)
        active_count = int(self.current_trace_progress) % (total_cells + 20)  # Add pause
        
        self.active_cells.clear()
        cell_count = 0
        for trace in self.traces:
            for x, y in trace:
                if cell_count < active_count:
                    self.active_cells.add((x, y))
                cell_count += 1
        
        # Render
        display_grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw traces
        for trace_idx, trace in enumerate(self.traces):
            for i, (x, y) in enumerate(trace):
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    # Determine trace character based on connections
                    if i == 0 or i == len(trace) - 1:
                        continue  # Skip nodes
                    
                    prev_x, prev_y = trace[i-1] if i > 0 else (x, y)
                    next_x, next_y = trace[i+1] if i < len(trace)-1 else (x, y)
                    
                    # Choose appropriate character
                    if prev_x == next_x:  # Vertical
                        char = '│'
                    elif prev_y == next_y:  # Horizontal
                        char = '─'
                    elif (prev_x < x and next_y < y) or (prev_y < y and next_x < x):
                        char = '┌'
                    elif (prev_x > x and next_y < y) or (prev_y < y and next_x > x):
                        char = '┐'
                    elif (prev_x < x and next_y > y) or (prev_y > y and next_x < x):
                        char = '└'
                    else:
                        char = '┘'
                    
                    display_grid[y][x] = char
                    styles[y][x] = self.active_color if (x, y) in self.active_cells else self.inactive_color
        
        # Draw nodes
        for x, y, char in self.nodes:
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                display_grid[y][x] = char
                styles[y][x] = self.node_color
        
        # Draw title
        if self.title:
            title_y = 1
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    display_grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = self.display_height - 2
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    display_grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = display_grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)