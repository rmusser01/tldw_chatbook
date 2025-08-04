"""GlitchReveal splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("glitch_reveal")
class GlitchRevealEffect(BaseEffect):
    """Reveals content by starting glitchy and becoming clear over time."""

    def __init__(
        self,
        parent_widget: Any,
        content: str, # The clear, final content
        duration: float = 2.0, # Total duration of the reveal effect
        glitch_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?",
        start_intensity: float = 0.8, # Initial glitch intensity (0.0 to 1.0)
        end_intensity: float = 0.0,   # Final glitch intensity
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.clear_content = content
        self.duration = duration
        self.glitch_chars = glitch_chars
        self.start_intensity = start_intensity
        self.end_intensity = end_intensity

        self.lines = self.clear_content.strip().split('\n')

    def update(self) -> Optional[str]:
        elapsed_time = time.time() - self.start_time
        progress = min(1.0, elapsed_time / self.duration) # Normalized time (0 to 1)

        # Intensity decreases over time (linear interpolation)
        current_intensity = self.start_intensity + (self.end_intensity - self.start_intensity) * progress
        # Could use an easing function for non-linear change in intensity.

        if current_intensity <= 0.01: # Effectively clear
            return self.clear_content.replace('[',r'\[')


        glitched_lines = []
        for line_idx, line_text in enumerate(self.lines):
            glitched_line_chars = list(line_text)
            for char_idx in range(len(glitched_line_chars)):
                if random.random() < current_intensity:
                    # Chance to replace char, or shift it, or change color
                    if random.random() < 0.7: # Replace char
                        glitched_line_chars[char_idx] = random.choice(self.glitch_chars)
                    # Could add other glitch types like small offsets or color shifts here
            glitched_lines.append("".join(glitched_line_chars))

        # Basic styling for glitched parts - can be enhanced
        output_lines = []
        for line in glitched_lines:
            escaped_line = line.replace('[', r'\[')
            # Randomly apply a "glitchy" color style to some parts
            if random.random() < current_intensity * 0.5: # More styling when more intense
                style = random.choice(["bold red", "bold blue", "bold yellow", "bold magenta"])
                output_lines.append(f"[{style}]{escaped_line}[/{style}]")
            else:
                output_lines.append(escaped_line) # Rely on card's base style

        return "\n".join(output_lines)