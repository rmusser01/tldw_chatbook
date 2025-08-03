"""ChaoticTypewriter splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("chaotic_typewriter")
class ChaoticTypewriterEffect(BaseEffect):
    """Multiple ghost typewriters typing at different speeds."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "",
        width: int = 80,
        height: int = 24,
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        
        # Create multiple typewriter instances
        self.typewriters = []
        num_typewriters = 5
        
        for i in range(num_typewriters):
            self.typewriters.append({
                'x': random.randint(5, width - len(title) - 5),
                'y': random.randint(2, height - 4),
                'speed': random.uniform(0.02, 0.08),
                'progress': 0,
                'last_update': time.time(),
                'color': random.choice([
                    'rgb(100,100,100)',
                    'rgb(150,150,150)',
                    'rgb(80,80,80)',
                    'rgb(120,120,120)',
                    'rgb(90,90,90)'
                ]),
                'drift_x': random.uniform(-0.5, 0.5),
                'drift_y': random.uniform(-0.2, 0.2),
                'text': title if i < 3 else subtitle  # Some type title, some subtitle
            })
        
        # Final positions
        self.final_title_x = (width - len(title)) // 2
        self.final_title_y = height // 2 - 1
        self.final_subtitle_x = (width - len(subtitle)) // 2
        self.final_subtitle_y = height // 2 + 1
        
        self.convergence_start = 2.0  # Start converging after 2 seconds
        self.convergence_duration = 1.5
    
    def update(self) -> Optional[str]:
        """Update chaotic typewriter effect."""
        elapsed = time.time() - self.start_time
        current_time = time.time()
        
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Update each typewriter
        for tw in self.typewriters:
            # Update typing progress
            if current_time - tw['last_update'] > tw['speed']:
                tw['progress'] = min(tw['progress'] + 1, len(tw['text']))
                tw['last_update'] = current_time
            
            # Update position with drift
            if elapsed < self.convergence_start:
                tw['x'] += tw['drift_x']
                tw['y'] += tw['drift_y']
                
                # Bounce off edges
                if tw['x'] < 0 or tw['x'] > self.width - len(tw['text']):
                    tw['drift_x'] *= -1
                if tw['y'] < 0 or tw['y'] > self.height - 1:
                    tw['drift_y'] *= -1
                
                tw['x'] = max(0, min(self.width - len(tw['text']), tw['x']))
                tw['y'] = max(0, min(self.height - 1, tw['y']))
            else:
                # Converge to final position
                convergence_progress = min(1.0, (elapsed - self.convergence_start) / self.convergence_duration)
                
                if tw['text'] == self.title:
                    target_x = self.final_title_x
                    target_y = self.final_title_y
                else:
                    target_x = self.final_subtitle_x
                    target_y = self.final_subtitle_y
                
                tw['x'] = tw['x'] + (target_x - tw['x']) * convergence_progress * 0.1
                tw['y'] = tw['y'] + (target_y - tw['y']) * convergence_progress * 0.1
            
            # Draw typewriter text
            x = int(tw['x'])
            y = int(tw['y'])
            text_to_show = tw['text'][:int(tw['progress'])]
            
            for i, char in enumerate(text_to_show):
                if 0 <= x + i < self.width and 0 <= y < self.height:
                    # During convergence, fade ghost typewriters
                    if elapsed > self.convergence_start:
                        convergence_progress = min(1.0, (elapsed - self.convergence_start) / self.convergence_duration)
                        if (tw['text'] == self.title and abs(y - self.final_title_y) < 1 and abs(x - self.final_title_x) < 1) or \
                           (tw['text'] == self.subtitle and abs(y - self.final_subtitle_y) < 1 and abs(x - self.final_subtitle_x) < 1):
                            # This is the final position
                            grid[y][x + i] = char
                            style_grid[y][x + i] = 'bold white' if tw['text'] == self.title else 'cyan'
                        else:
                            # Ghost typewriter fading out
                            if random.random() > convergence_progress:
                                grid[y][x + i] = char
                                style_grid[y][x + i] = tw['color']
                    else:
                        grid[y][x + i] = char
                        style_grid[y][x + i] = tw['color']
            
            # Typewriter cursor
            if tw['progress'] < len(tw['text']) and int(current_time * 2) % 2 == 0:
                cursor_x = x + int(tw['progress'])
                if 0 <= cursor_x < self.width and 0 <= y < self.height:
                    grid[y][cursor_x] = '▌'
                    style_grid[y][cursor_x] = tw['color']
        
        # Convert to string
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                char = grid[y][x]
                style = style_grid[y][x]
                if style:
                    line += f"[{style}]{char}[/{style.split()[0]}]"
                else:
                    line += char
            lines.append(line)
        return '\n'.join(lines)