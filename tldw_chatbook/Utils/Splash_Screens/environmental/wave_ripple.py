"""WaveRipple splash screen effect."""

import math
import time
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect


@register_effect("wave_ripple")
class WaveRippleEffect(BaseEffect):
    """Creates expanding ripple patterns from the center outward, like dropping a stone in water."""
    
    @dataclass
    class Wave:
        radius: float
        max_radius: float
        amplitude: float  # Height of the wave
        speed: float
        color_index: int
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Rippling through data...",
        width: int = 80,
        height: int = 24,
        wave_speed: float = 10.0,  # Pixels per second
        wave_spawn_rate: float = 1.0,  # New waves per second
        max_waves: int = 5,
        wave_chars: str = "·∙○◯◉●",
        wave_colors: List[str] = ["dim cyan", "cyan", "bright_cyan", "white", "cyan", "dim cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.wave_speed = wave_speed
        self.wave_spawn_rate = wave_spawn_rate
        self.max_waves = max_waves
        self.wave_chars = wave_chars
        self.wave_colors = wave_colors
        self.title_style = title_style
        
        self.waves: List[WaveRippleEffect.Wave] = []
        self.time_since_last_wave = 0.0
        self.last_update_time = time.time()
        
        # Center point for ripples
        self.center_x = width / 2
        self.center_y = height / 2
        
    def _spawn_wave(self):
        """Spawn a new wave from the center."""
        if len(self.waves) < self.max_waves:
            max_radius = math.sqrt((self.display_width/2)**2 + (self.display_height/2)**2)
            self.waves.append(WaveRippleEffect.Wave(
                radius=0.0,
                max_radius=max_radius,
                amplitude=1.0,
                speed=self.wave_speed,
                color_index=0
            ))
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Spawn new waves
        self.time_since_last_wave += delta_time
        if self.time_since_last_wave >= 1.0 / self.wave_spawn_rate:
            self._spawn_wave()
            self.time_since_last_wave = 0.0
        
        # Update existing waves
        active_waves = []
        for wave in self.waves:
            wave.radius += wave.speed * delta_time
            if wave.radius < wave.max_radius:
                # Update amplitude (fade out as it expands)
                wave.amplitude = max(0, 1.0 - (wave.radius / wave.max_radius))
                # Update color index based on radius
                wave.color_index = int((wave.radius / wave.max_radius) * (len(self.wave_colors) - 1))
                active_waves.append(wave)
        self.waves = active_waves
        
        # Render waves
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw waves (older/larger ones first)
        for wave in sorted(self.waves, key=lambda w: w.radius, reverse=True):
            # Draw circle at current radius
            num_points = int(2 * math.pi * wave.radius) + 1
            if num_points < 4:
                num_points = 4
                
            for i in range(num_points):
                angle = (2 * math.pi * i) / num_points
                x = int(self.center_x + wave.radius * math.cos(angle))
                y = int(self.center_y + wave.radius * math.sin(angle) * 0.5)  # Aspect ratio correction
                
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    # Choose character based on wave amplitude
                    char_index = int((1.0 - wave.amplitude) * (len(self.wave_chars) - 1))
                    char_index = max(0, min(char_index, len(self.wave_chars) - 1))
                    
                    grid[y][x] = self.wave_chars[char_index]
                    styles[y][x] = self.wave_colors[wave.color_index]
        
        # Overlay title and subtitle
        if self.title:
            title_y = int(self.center_y) - 2
            title_x = int((self.display_width - len(self.title)) / 2)
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        if self.subtitle:
            subtitle_y = int(self.center_y) + 1
            subtitle_x = int((self.display_width - len(self.subtitle)) / 2)
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = grid[y][x]
                style = styles[y][x]
                
                if style:
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)