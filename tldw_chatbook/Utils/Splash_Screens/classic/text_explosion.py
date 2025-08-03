"""TextExplosion splash screen effect."""

import math
import random
import time
from typing import Optional, Any, List, Tuple
from dataclasses import dataclass

from ..base_effect import BaseEffect, register_effect


@register_effect("text_explosion")
class TextExplosionEffect(BaseEffect):
    """Animates characters of a text to explode outwards or implode inwards."""

    @dataclass
    class AnimatedChar:
        char: str
        origin_x: float # Final x in assembled text
        origin_y: float # Final y in assembled text
        current_x: float
        current_y: float
        vx: float # velocity for explosion (if random start)
        vy: float # velocity for explosion
        style: str = "bold white"

    def __init__(
        self,
        parent_widget: Any,
        text: str = "EXPLODE!",
        effect_type: str = "explode", # "explode" or "implode"
        duration: float = 1.5,
        width: int = 80, # Display area
        height: int = 24,
        char_style: str = "bold yellow",
        particle_spread: float = 30.0, # How far particles spread for explosion/implosion start
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.target_text = text
        self.effect_type = effect_type
        self.duration = duration
        self.display_width = width
        self.display_height = height
        self.char_style = char_style # Can be a list of styles too
        self.particle_spread = particle_spread

        self.chars: List[TextExplosionEffect.AnimatedChar] = []
        self._prepare_chars()

    def _prepare_chars(self):
        # Simple centered text for now
        text_width = len(self.target_text)
        start_x = (self.display_width - text_width) / 2.0
        origin_y = self.display_height / 2.0

        for i, char_val in enumerate(self.target_text):
            ox = start_x + i
            oy = origin_y

            if self.effect_type == "explode":
                # Start at origin, will move outwards
                cx, cy = ox, oy
                # Velocity is random, pointing outwards from text's rough center
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(self.particle_spread*0.5, self.particle_spread*1.5) / self.duration
                vx, vy = math.cos(angle) * speed, math.sin(angle) * speed * 0.5 # Y velocity halved for aspect
            else: # implode
                # Start at random spread-out position, move towards origin
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(self.particle_spread * 0.5, self.particle_spread)
                cx = ox + math.cos(angle) * dist
                cy = oy + math.sin(angle) * dist * 0.5 # Y spread halved
                vx, vy = 0,0 # Not used for velocity in implode, calculated by interpolation

            self.chars.append(TextExplosionEffect.AnimatedChar(
                char=char_val, origin_x=ox, origin_y=oy, current_x=cx, current_y=cy, vx=vx, vy=vy, style=self.char_style
            ))

    def update(self) -> Optional[str]:
        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration) # Normalized time 0 to 1

        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        # Store (char, style) to handle multiple chars on same spot (last one wins, or brightest)
        styled_grid: Dict[Tuple[int,int], Tuple[str,str]] = {}


        for achar in self.chars:
            if self.effect_type == "explode":
                # Move based on initial velocity, progress acting as time factor
                # This makes them fly out at constant speed after "explosion"
                # A better explosion might have initial burst then slowdown.
                # For now, constant velocity after start.
                if progress > 0.05 : # Small delay before explosion starts or to ensure movement
                    achar.current_x += achar.vx * progress * 20 # Scale velocity effect by progress and arbitrary factor
                    achar.current_y += achar.vy * progress * 20
            else: # implode
                # Interpolate from start (current_x, current_y at progress=0) to origin_x, origin_y
                # Initial current_x/y were set in _prepare_chars
                # We need to store the very initial random positions
                if not hasattr(achar, 'initial_x'): # Store initial random pos if not already
                    achar.initial_x = achar.current_x
                    achar.initial_y = achar.current_y

                # Interpolate based on progress. At progress=0, use initial. At progress=1, use origin.
                achar.current_x = achar.initial_x + (achar.origin_x - achar.initial_x) * progress
                achar.current_y = achar.initial_y + (achar.origin_y - achar.initial_y) * progress

            ix, iy = int(achar.current_x), int(achar.current_y)
            if 0 <= ix < self.display_width and 0 <= iy < self.display_height:
                # Simplistic: last char to land on a cell wins
                styled_grid[(ix,iy)] = (achar.char, achar.style)

        # Render styled_grid to output lines
        output_lines = []
        for r_idx in range(self.display_height):
            line_segments = []
            for c_idx in range(self.display_width):
                if (c_idx, r_idx) in styled_grid:
                    char, style = styled_grid[(c_idx, r_idx)]
                    escaped_char = char.replace('[', ESCAPED_OPEN_BRACKET)
                    line_segments.append(f"[{style}]{escaped_char}[/{style}]")
                else:
                    line_segments.append(' ')
            output_lines.append("".join(line_segments))

        return "\n".join(output_lines)