"""NeuralNetwork splash screen effect."""
import time

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("neural_network")
class NeuralNetworkEffect(BaseEffect):
    """Neural network visualization with nodes and connections."""
    
    def __init__(self, parent, title="TLDW Chatbook", subtitle="", width=80, height=24, speed=0.1, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.subtitle = subtitle
        self.nodes = []
        self.connections = []
        self.activation_wave = 0
        
        # Create network structure
        layers = [3, 5, 4, 5, 3]  # Nodes per layer
        layer_spacing = self.width // (len(layers) + 1)
        
        for layer_idx, node_count in enumerate(layers):
            x = layer_spacing * (layer_idx + 1)
            layer_nodes = []
            node_spacing = self.height // (node_count + 1)
            
            for node_idx in range(node_count):
                y = node_spacing * (node_idx + 1)
                node = {
                    'x': x,
                    'y': y,
                    'layer': layer_idx,
                    'activation': 0.0,
                    'id': f"L{layer_idx}N{node_idx}"
                }
                layer_nodes.append(node)
                self.nodes.append(node)
            
            # Create connections to previous layer
            if layer_idx > 0:
                prev_layer_nodes = [n for n in self.nodes if n['layer'] == layer_idx - 1]
                for node in layer_nodes:
                    for prev_node in prev_layer_nodes:
                        if random.random() > 0.3:  # 70% connection probability
                            self.connections.append({
                                'from': prev_node,
                                'to': node,
                                'strength': random.random()
                            })
    
    def update(self) -> Optional[str]:
        """Update neural network animation."""
        elapsed_time = time.time() - self.start_time
        self.activation_wave = (self.activation_wave + elapsed_time * 2) % (len(set(n['layer'] for n in self.nodes)) + 2)
        
        # Update node activations
        for node in self.nodes:
            if abs(node['layer'] - self.activation_wave) < 1:
                node['activation'] = min(1.0, node['activation'] + elapsed_time * 3)
            else:
                node['activation'] = max(0.0, node['activation'] - elapsed_time * 2)
        
        # Return the rendered content
        return self.render()
    
    def render(self):
        """Render the neural network."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw connections
        for conn in self.connections:
            if conn['from']['activation'] > 0.1 or conn['to']['activation'] > 0.1:
                strength = (conn['from']['activation'] + conn['to']['activation']) / 2 * conn['strength']
                self._draw_line(grid, style_grid, 
                               conn['from']['x'], conn['from']['y'],
                               conn['to']['x'], conn['to']['y'],
                               strength)
        
        # Draw nodes
        for node in self.nodes:
            x, y = node['x'], node['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                if node['activation'] > 0.8:
                    grid[y][x] = '●'
                    style_grid[y][x] = 'bold cyan'
                elif node['activation'] > 0.4:
                    grid[y][x] = '◉'
                    style_grid[y][x] = 'cyan'
                elif node['activation'] > 0.1:
                    grid[y][x] = '○'
                    style_grid[y][x] = 'dim cyan'
                else:
                    grid[y][x] = '·'
                    style_grid[y][x] = 'dim white'
        
        # Add title when network is active
        if self.activation_wave > 1:
            self._add_centered_text(grid, style_grid, self.title, self.height // 2 - 1, 'bold white')
            if self.subtitle:
                self._add_centered_text(grid, style_grid, self.subtitle, self.height // 2 + 1, 'white')
        
        return self._grid_to_string(grid, style_grid)
    
    def _draw_line(self, grid, style_grid, x1, y1, x2, y2, strength):
        """Draw a line between two points."""
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps == 0:
            return
        
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                if grid[y][x] == ' ':
                    if strength > 0.7:
                        grid[y][x] = '═' if abs(x2 - x1) > abs(y2 - y1) else '║'
                        style_grid[y][x] = 'bold blue'
                    elif strength > 0.3:
                        grid[y][x] = '─' if abs(x2 - x1) > abs(y2 - y1) else '│'
                        style_grid[y][x] = 'blue'
                    else:
                        grid[y][x] = '·'
                        style_grid[y][x] = 'dim blue'