"""Blink splash screen effect."""

from rich.style import Style
import time
from typing import Optional, Any, List, Tuple
from rich.text import Text

from ..base_effect import BaseEffect, register_effect


@register_effect("blink")
class BlinkEffect(BaseEffect):
    """Makes specified parts of the text blink."""

    def __init__(
        self,
        parent_widget: Any,
        content: str,
        blink_speed: float = 0.5,  # Time for one state (on or off)
        blink_targets: Optional[List[str]] = None, # List of exact strings to blink
        blink_style_on: str = "default", # Style when text is visible
        blink_style_off: str = "dim",  # Style when text is "off" (e.g., dimmed or hidden via color)
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.original_content = content
        self.blink_speed = blink_speed
        self.blink_targets = blink_targets if blink_targets else []
        self.blink_style_on = blink_style_on # Not actively used if "default" means use card's base style
        self.blink_style_off = blink_style_off
        self._is_on = True
        self._last_blink_time = time.time()

    def update(self) -> Optional[str]:
        current_time = time.time()
        if current_time - self._last_blink_time >= self.blink_speed:
            self._is_on = not self._is_on
            self._last_blink_time = current_time

        # This is a simplified blink effect. For complex Rich text, direct string
        # manipulation of styled text is tricky. This version applies a style to target
        # strings when they should be "off" or replaces them.

        # Start with the original content, pre-escaped for Rich tags.
        # This assumes the card's main style will handle the "on" state appearance.
        # The effect focuses on altering the "off" state or replacing text.

        output_text = Text.from_markup(self.original_content.replace('[', r'\['))

        if not self._is_on:
            for target_text in self.blink_targets:
                # Find all occurrences of target_text and apply blink_style_off or hide
                start_index = 0
                while True:
                    try:
                        # Search in the plain string version of the Text object
                        found_pos = output_text.plain.find(target_text, start_index)
                        if found_pos == -1:
                            break

                        if self.blink_style_off == "hidden":
                            # Replace with spaces
                            output_text.plain = output_text.plain[:found_pos] + ' ' * len(target_text) + output_text.plain[found_pos+len(target_text):]
                            # This modification of .plain is a bit of a hack.
                            # A more robust way would be to reconstruct the Text object or use Text.replace.
                            # For now, let's rebuild the text object to ensure spans are cleared.
                            current_plain = output_text.plain
                            output_text = Text(current_plain) # Re-create to clear old spans over modified region
                        else:
                            # Apply style
                            output_text.stylize(self.blink_style_off, start=found_pos, end=found_pos + len(target_text))

                        start_index = found_pos + len(target_text)
                    except ValueError: # Should not happen with plain.find
                        break
        # If _is_on, the text remains as is, relying on the Static widget's base style.
        # If blink_style_on was not "default", one would apply it here to the targets.

        return output_text.markup # Return the Rich markup string