"""RubiksCube splash screen effect."""
import time

import random
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("rubiks_cube")
class RubiksCubeEffect(BaseEffect):
    """3D ASCII Rubik's cube solving itself."""
    
    def __init__(self, parent, title="TLDW", width=80, height=24, speed=0.5, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.rotation_x = 0
        self.rotation_y = 0
        self.solve_progress = 0
        self.current_move = None
        self.move_progress = 0
        
        # Simplified cube representation
        self.faces = {
            'F': [['R' for _ in range(3)] for _ in range(3)],  # Front - Red
            'B': [['O' for _ in range(3)] for _ in range(3)],  # Back - Orange
            'U': [['W' for _ in range(3)] for _ in range(3)],  # Up - White
            'D': [['Y' for _ in range(3)] for _ in range(3)],  # Down - Yellow
            'L': [['G' for _ in range(3)] for _ in range(3)],  # Left - Green
            'R': [['B' for _ in range(3)] for _ in range(3)]   # Right - Blue
        }
        
        # Scramble cube
        self._scramble()
    
    def _scramble(self):
        """Scramble the cube."""
        moves = ['F', 'B', 'U', 'D', 'L', 'R']
        for _ in range(20):
            self._rotate_face(random.choice(moves))
    
    def _rotate_face(self, face):
        """Rotate a face 90 degrees clockwise."""
        # Simplified rotation - just shuffle colors
        if face in self.faces:
            # Rotate the face itself
            face_data = self.faces[face]
            rotated = [[face_data[2-j][i] for j in range(3)] for i in range(3)]
            self.faces[face] = rotated
    
    def update(self) -> Optional[str]:
        """Update cube animation."""
        elapsed_time = time.time() - self.start_time
        # Rotate cube for 3D effect
        self.rotation_y += elapsed_time * 0.5
        
        # Solve animation
        self.solve_progress += elapsed_time * 0.1
        
        # Perform moves
        if not self.current_move and random.random() < 0.3:
            self.current_move = random.choice(['F', 'B', 'U', 'D', 'L', 'R'])
            self.move_progress = 0
        
        if self.current_move:
            self.move_progress += elapsed_time * 3
            if self.move_progress >= 1.0:
                self._rotate_face(self.current_move)
                self.current_move = None
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render 3D Rubik's cube."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Draw isometric cube
        # Top face (U)
        top_start_x = center_x - 6
        top_start_y = center_y - 8
        
        for i in range(3):
            for j in range(3):
                x = top_start_x + i * 4 + j * 2
                y = top_start_y + i * 2 - j
                if 0 <= x < self.width and 0 <= y < self.height:
                    color = self.faces['U'][i][j]
                    grid[y][x] = '▄'
                    grid[y][x + 1] = '▄'
                    style_grid[y][x] = style_grid[y][x + 1] = self._get_color_style(color)
        
        # Front face (F)
        front_start_x = center_x - 6
        front_start_y = center_y - 2
        
        for i in range(3):
            for j in range(3):
                x = front_start_x + j * 4
                y = front_start_y + i * 2
                if 0 <= x < self.width and 0 <= y < self.height:
                    color = self.faces['F'][i][j]
                    grid[y][x] = '█'
                    grid[y][x + 1] = '█'
                    if y + 1 < self.height:
                        grid[y + 1][x] = '█'
                        grid[y + 1][x + 1] = '█'
                    style_grid[y][x] = style_grid[y][x + 1] = self._get_color_style(color)
                    if y + 1 < self.height:
                        style_grid[y + 1][x] = style_grid[y + 1][x + 1] = self._get_color_style(color)
        
        # Right face (R) - partial view
        right_start_x = center_x + 6
        right_start_y = center_y - 5
        
        for i in range(3):
            for j in range(3):
                x = right_start_x + j * 2
                y = right_start_y + i * 2 + j
                if 0 <= x < self.width and 0 <= y < self.height:
                    color = self.faces['R'][i][j]
                    grid[y][x] = '▐'
                    style_grid[y][x] = self._get_color_style(color)
        
        # Add title when cube is solving
        if self.solve_progress > 0.3:
            self._add_centered_text(grid, style_grid, self.title, self.height - 3, 'bold white')
            if self.solve_progress > 0.7:
                self._add_centered_text(grid, style_grid, "SOLVED!", self.height - 1, 'bold green')
        
        return self._grid_to_string(grid, style_grid)
    
    def _get_color_style(self, color_char):
        """Get style for cube colors."""
        color_map = {
            'R': 'red',
            'O': 'rgb(255,165,0)',  # Orange
            'W': 'white',
            'Y': 'yellow',
            'G': 'green',
            'B': 'blue'
        }
        return f"bold {color_map.get(color_char, 'white')}"