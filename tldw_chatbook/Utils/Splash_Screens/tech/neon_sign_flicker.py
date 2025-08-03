"""NeonSignFlicker splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("neon_sign_flicker")
class NeonSignFlickerEffect(BaseEffect):
    """Vintage neon sign with realistic flicker effect."""
    
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
        
        # Neon sign design
        self.sign_text = ["TLDW", "CHATBOOK"]
        self.sign_border = True
        
        # Flicker states for each character
        self.char_states = {}
        self.flicker_probability = 0.05
        self.startup_progress = 0.0
        
        # Glow effect characters
        self.glow_chars = {
            'dim': '░',
            'medium': '▒',
            'bright': '▓'
        }
        
        # Electrical buzz visualization
        self.buzz_particles = []
        
    def create_neon_letter(self, letter):
        """Create neon-style version of a letter."""
        # Double-line effect for neon tubes
        neon_map = {
            'T': ["╔═══╗", "  ║  ", "  ║  "],
            'L': ["║    ", "║    ", "╚═══ "],
            'D': ["╔══╗ ", "║  ║ ", "╚══╝ "],
            'W': ["║   ║", "║ ║ ║", "╚═╩═╝"],
            'C': ["╔═══ ", "║    ", "╚═══ "],
            'H': ["║   ║", "╠═══╣", "║   ║"],
            'A': ["╔═══╗", "╠═══╣", "║   ║"],
            'B': ["╔══╗ ", "╠══╣ ", "╚══╝ "],
            'O': ["╔═══╗", "║   ║", "╚═══╝"],
            'K': ["║  ╱ ", "╠═╱  ", "║ ╲  "]
        }
        return neon_map.get(letter, ["     ", "     ", "     "])
    
    def update(self) -> Optional[str]:
        """Update the neon sign animation."""
        elapsed = time.time() - self.start_time
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Update startup progress
        if elapsed < 2.0:
            self.startup_progress = elapsed / 2.0
        else:
            self.startup_progress = 1.0
        
        # Draw sign frame
        frame_width = 60
        frame_height = 14
        frame_x = (self.width - frame_width) // 2
        frame_y = 5
        
        if self.sign_border:
            # Top and bottom borders
            for x in range(frame_x, frame_x + frame_width):
                if x < self.width:
                    grid[frame_y][x] = '═'
                    grid[frame_y + frame_height][x] = '═'
            
            # Side borders
            for y in range(frame_y + 1, frame_y + frame_height):
                if y < self.height:
                    grid[y][frame_x] = '║'
                    grid[y][frame_x + frame_width - 1] = '║'
            
            # Corners
            grid[frame_y][frame_x] = '╔'
            grid[frame_y][frame_x + frame_width - 1] = '╗'
            grid[frame_y + frame_height][frame_x] = '╚'
            grid[frame_y + frame_height][frame_x + frame_width - 1] = '╝'
        
        # Draw neon text
        text_y = frame_y + 3
        
        for line_idx, line in enumerate(self.sign_text):
            # Calculate line position
            line_width = len(line) * 6
            line_x = (self.width - line_width) // 2
            line_y = text_y + line_idx * 5
            
            # Draw each letter
            for char_idx, char in enumerate(line):
                char_x = line_x + char_idx * 6
                char_key = f"{line_idx}_{char_idx}"
                
                # Initialize character state
                if char_key not in self.char_states:
                    self.char_states[char_key] = {
                        'on': False,
                        'brightness': 0.0,
                        'flicker_timer': 0
                    }
                
                state = self.char_states[char_key]
                
                # Update character state
                if self.startup_progress < 1.0:
                    # Startup sequence - letters turn on sequentially
                    turn_on_time = (line_idx * len(line) + char_idx) / (len(self.sign_text) * max(len(l) for l in self.sign_text))
                    if self.startup_progress > turn_on_time:
                        state['on'] = True
                        state['brightness'] = min(1.0, (self.startup_progress - turn_on_time) * 4)
                else:
                    # Normal operation with occasional flicker
                    if state['flicker_timer'] > 0:
                        state['flicker_timer'] -= 1
                        state['on'] = (state['flicker_timer'] % 2 == 0)
                    else:
                        state['on'] = True
                        # Random flicker
                        if random.random() < self.flicker_probability:
                            state['flicker_timer'] = random.randint(3, 8)
                    
                    state['brightness'] = 1.0 if state['on'] else 0.2
                
                # Draw letter if on
                if state['brightness'] > 0:
                    letter_pattern = self.create_neon_letter(char)
                    for row_idx, row in enumerate(letter_pattern):
                        for col_idx, pixel in enumerate(row):
                            x = char_x + col_idx
                            y = line_y + row_idx
                            if 0 <= x < self.width and 0 <= y < self.height and pixel != ' ':
                                grid[y][x] = pixel
                                
                                # Add glow effect around bright letters
                                if state['brightness'] > 0.8:
                                    for dx in [-1, 0, 1]:
                                        for dy in [-1, 0, 1]:
                                            gx, gy = x + dx, y + dy
                                            if (0 <= gx < self.width and 0 <= gy < self.height and 
                                                grid[gy][gx] == ' ' and random.random() < 0.3):
                                                grid[gy][gx] = self.glow_chars['dim']
        
        # Add electrical effects
        if self.startup_progress >= 1.0 and random.random() < 0.1:
            # Spawn buzz particle
            self.buzz_particles.append({
                'x': random.randint(frame_x - 2, frame_x + frame_width + 2),
                'y': random.randint(frame_y - 2, frame_y + frame_height + 2),
                'life': random.randint(3, 6),
                'char': random.choice(['*', '×', '+'])
            })
        
        # Update and draw buzz particles
        particles_to_remove = []
        for particle in self.buzz_particles:
            particle['life'] -= 1
            if particle['life'] <= 0:
                particles_to_remove.append(particle)
            else:
                x, y = particle['x'], particle['y']
                if 0 <= x < self.width and 0 <= y < self.height:
                    grid[y][x] = particle['char']
        
        for particle in particles_to_remove:
            self.buzz_particles.remove(particle)
        
        # Draw subtitle
        if self.startup_progress >= 1.0:
            subtitle = "◆ OPEN 24/7 ◆"
        else:
            subtitle = "◆ STARTING UP ◆"
        
        subtitle_x = (self.width - len(subtitle)) // 2
        subtitle_y = frame_y + frame_height + 2
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width and subtitle_y < self.height:
                grid[subtitle_y][subtitle_x + i] = char
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if char in '═║╔╗╚╝':  # Frame
                    line += f"[dim white]{char}[/dim white]"
                elif char in '╠╣╱╲':  # Neon letters
                    # Check brightness of corresponding character
                    brightness_found = False
                    for char_key, state in self.char_states.items():
                        if state['brightness'] > 0.8:
                            line += f"[bold magenta]{char}[/bold magenta]"
                            brightness_found = True
                            break
                    if not brightness_found:
                        line += f"[dim magenta]{char}[/dim magenta]"
                elif char in '░▒▓':  # Glow
                    line += f"[dim magenta]{char}[/dim magenta]"
                elif char in '*×+':  # Electrical effects
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char == '◆':  # Decorative
                    line += f"[yellow]{char}[/yellow]"
                elif y == subtitle_y:  # Subtitle
                    line += f"[cyan]{char}[/cyan]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)