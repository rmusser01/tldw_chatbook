"""Mining splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("mining")
class MiningEffect(BaseEffect):
    """Creates a Dwarf Fortress-style mining/digging animation."""
    
    def __init__(
        self,
        parent_widget: Any,
        content: str,
        width: int = 80,
        height: int = 24,
        dig_speed: float = 0.8,  # How fast the mining progresses
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.content_lines = content.strip().split('\n')
        self.width = width
        self.height = height
        self.dig_speed = dig_speed
        
        # Mining characters progression: stone -> partially dug -> empty
        self.stone_chars = ['█', '▓', '▒', '░', ' ']
        self.stone_styles = ['rgb(100,60,20)', 'rgb(120,80,40)', 'rgb(140,100,60)', 'rgb(160,120,80)', 'white']
        
        # Initialize mining grid - start with all stone
        self.mining_grid = [[0 for _ in range(width)] for _ in range(height)]
        self.reveal_grid = [[False for _ in range(width)] for _ in range(height)]
        
        # Track mining progress
        self.mining_progress = 0.0
        self.total_cells = sum(len(line) for line in self.content_lines)
        
        # Pickaxe animation
        self.pickaxe_pos = [width // 2, height // 2]
        self.pickaxe_char = '⚒'
        self.pickaxe_style = 'bold yellow'
        
        # Sound effect characters
        self.sound_effects = ['*', '∆', '◊', '✦']
        self.sound_positions = []
        
    def update(self) -> Optional[str]:
        """Update mining animation."""
        elapsed = time.time() - self.start_time
        
        # Progress mining based on elapsed time
        target_progress = min(1.0, elapsed * self.dig_speed / 5.0)  # 5 seconds total
        
        # Mine out cells progressively
        if target_progress > self.mining_progress:
            self.mining_progress = target_progress
            self._advance_mining()
        
        # Update pickaxe position (move it around as if mining)
        self._update_pickaxe_position()
        
        # Add sound effects
        if random.random() < 0.3:  # 30% chance each frame
            self._add_sound_effect()
        
        # Decay sound effects
        self._update_sound_effects()
        
        return self._render_frame()
    
    def _advance_mining(self):
        """Advance the mining progress, revealing content."""
        # Calculate how many cells should be revealed
        cells_to_reveal = int(self.mining_progress * self.total_cells)
        
        # Reveal cells in a mining pattern (roughly center-out)
        revealed_count = 0
        center_y = len(self.content_lines) // 2
        
        for distance in range(max(self.width, self.height)):
            if revealed_count >= cells_to_reveal:
                break
                
            # Mine in expanding pattern from center
            for line_idx, line in enumerate(self.content_lines):
                # Check line_idx bounds
                if line_idx >= self.height:
                    break
                if abs(line_idx - center_y) <= distance:
                    for char_idx, char in enumerate(line):
                        if revealed_count >= cells_to_reveal:
                            break
                        # Check bounds to prevent IndexError
                        if char_idx < self.width and not self.reveal_grid[line_idx][char_idx] and char != ' ':
                            self.reveal_grid[line_idx][char_idx] = True
                            # Set mining stage (gradually dig through stone)
                            self.mining_grid[line_idx][char_idx] = min(4, int(self.mining_progress * 8))
                            revealed_count += 1
    
    def _update_pickaxe_position(self):
        """Move pickaxe around as if actively mining."""
        # Simple circular motion with some randomness
        angle = time.time() * 2.0 + random.uniform(-0.5, 0.5)
        center_x = self.width // 2
        center_y = self.height // 2
        
        radius = 3 + random.uniform(-1, 1)
        self.pickaxe_pos[0] = int(center_x + math.cos(angle) * radius)
        self.pickaxe_pos[1] = int(center_y + math.sin(angle) * radius * 0.5)  # Smaller vertical movement
        
        # Keep within bounds
        self.pickaxe_pos[0] = max(0, min(self.width - 1, self.pickaxe_pos[0]))
        self.pickaxe_pos[1] = max(0, min(self.height - 1, self.pickaxe_pos[1]))
    
    def _add_sound_effect(self):
        """Add a sound effect near the pickaxe."""
        if len(self.sound_positions) < 5:  # Limit number of effects
            effect_char = random.choice(self.sound_effects)
            # Place near pickaxe with some randomness
            x = self.pickaxe_pos[0] + random.randint(-2, 2)
            y = self.pickaxe_pos[1] + random.randint(-1, 1)
            
            if 0 <= x < self.width and 0 <= y < self.height:
                self.sound_positions.append({
                    'char': effect_char,
                    'x': x,
                    'y': y,
                    'life': 1.0,
                    'style': random.choice(['bold red', 'bold orange', 'bold yellow'])
                })
    
    def _update_sound_effects(self):
        """Update and decay sound effects."""
        for effect in self.sound_positions[:]:  # Copy list to modify while iterating
            effect['life'] -= 0.1
            if effect['life'] <= 0:
                self.sound_positions.remove(effect)
    
    def _render_frame(self) -> str:
        """Render the current mining frame."""
        # Start with empty grid
        display_grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [['' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw the stone/mining progress
        for y in range(min(len(self.content_lines), self.height)):
            line = self.content_lines[y]
            for x in range(min(len(line), self.width)):
                char = line[x]
                
                if char != ' ':
                    if self.reveal_grid[y][x]:
                        # Cell is being mined - show mining progress
                        stage = self.mining_grid[y][x]
                        if stage >= len(self.stone_chars) - 1:
                            # Fully mined - show original character
                            display_grid[y][x] = char
                            style_grid[y][x] = 'white'
                        else:
                            # Show mining progress
                            display_grid[y][x] = self.stone_chars[stage]
                            style_grid[y][x] = self.stone_styles[stage]
                    else:
                        # Unmined stone
                        display_grid[y][x] = self.stone_chars[0]
                        style_grid[y][x] = self.stone_styles[0]
        
        # Draw pickaxe
        px, py = self.pickaxe_pos
        if 0 <= px < self.width and 0 <= py < self.height:
            display_grid[py][px] = self.pickaxe_char
            style_grid[py][px] = self.pickaxe_style
        
        # Draw sound effects
        for effect in self.sound_positions:
            x, y = effect['x'], effect['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                if effect['life'] > 0.5:  # Only show if still fresh
                    display_grid[y][x] = effect['char']
                    style_grid[y][x] = effect['style']
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.height):
            line_segments = []
            for x in range(self.width):
                char = display_grid[y][x]
                style = style_grid[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)