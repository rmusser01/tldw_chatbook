"""FractalZoom splash screen effect."""
import time

from rich.color import Color
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("fractal_zoom")
class FractalZoomEffect(BaseEffect):
    """Mandelbrot fractal zoom effect."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.zoom = 1.0
        self.center_x = -0.5
        self.center_y = 0.0
        self.max_iter = 50
        self.chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    
    def update(self) -> Optional[str]:
        """Update fractal zoom."""
        elapsed_time = time.time() - self.start_time
        self.zoom *= 1.0 + elapsed_time * 0.5
        
        # Slowly drift center
        self.center_x += elapsed_time * 0.01
        
        # Return the rendered content
        return self.render()
    
    def _mandelbrot(self, c_real, c_imag):
        """Calculate Mandelbrot iteration count."""
        z_real, z_imag = 0, 0
        
        for i in range(self.max_iter):
            if z_real * z_real + z_imag * z_imag > 4:
                return i
            
            z_real_new = z_real * z_real - z_imag * z_imag + c_real
            z_imag = 2 * z_real * z_imag + c_imag
            z_real = z_real_new
        
        return self.max_iter
    
    def render(self):
        """Render fractal zoom."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Calculate fractal
        for y in range(self.height):
            for x in range(self.width):
                # Map pixel to complex plane
                real = (x - self.width / 2) / (self.zoom * self.width / 4) + self.center_x
                imag = (y - self.height / 2) / (self.zoom * self.height / 4) + self.center_y
                
                # Calculate iterations
                iterations = self._mandelbrot(real, imag)
                
                # Map to character
                if iterations == self.max_iter:
                    grid[y][x] = ' '
                else:
                    char_index = iterations % len(self.chars)
                    grid[y][x] = self.chars[char_index]
                    
                    # Color based on iteration count
                    if iterations < 10:
                        style_grid[y][x] = 'blue'
                    elif iterations < 20:
                        style_grid[y][x] = 'cyan'
                    elif iterations < 30:
                        style_grid[y][x] = 'green'
                    elif iterations < 40:
                        style_grid[y][x] = 'yellow'
                    else:
                        style_grid[y][x] = 'red'
        
        # Add title overlay
        if self.zoom > 3:
            self._add_centered_text(grid, style_grid, self.title, self.height // 2, 'bold white on black')
        
        return self._grid_to_string(grid, style_grid)