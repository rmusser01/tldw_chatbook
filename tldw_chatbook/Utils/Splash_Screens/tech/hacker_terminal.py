"""HackerTerminal splash screen effect."""
import time

from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("hacker_terminal")
class HackerTerminalEffect(BaseEffect):
    """Hacking simulation with terminal commands."""
    
    def __init__(self, parent, title="TLDW Chatbook", width=80, height=24, speed=0.05, **kwargs):
        super().__init__(parent, width=width, height=height, speed=speed)
        self.width = width
        self.height = height
        self.speed = speed
        self.title = title
        self.terminal_lines = []
        self.current_command = ""
        self.typing_progress = 0
        self.phase = "connecting"  # connecting, scanning, cracking, granted
        self.phase_start_time = 0
        
        # Hacking sequence
        self.commands = [
            ("$ ssh root@mainframe.tldw.ai", "Connecting to mainframe..."),
            ("$ nmap -sS -p- 192.168.1.1", "Scanning ports..."),
            ("$ ./exploit.sh --target=firewall", "Bypassing firewall..."),
            ("$ hashcat -m 0 -a 0 hashes.txt wordlist.txt", "Cracking passwords..."),
            ("$ sudo access --grant-all", "Escalating privileges..."),
        ]
        self.current_command_index = 0
    
    def update(self) -> Optional[str]:
        """Update hacking simulation."""
        elapsed_time = time.time() - self.start_time
        self.phase_start_time += elapsed_time
        
        # Type current command
        if self.current_command_index < len(self.commands):
            cmd, _ = self.commands[self.current_command_index]
            if self.typing_progress < len(cmd):
                self.typing_progress += elapsed_time * 30  # Typing speed
                self.current_command = cmd[:int(self.typing_progress)]
            else:
                # Command complete, add to terminal
                if self.current_command and self.current_command not in [line['text'] for line in self.terminal_lines]:
                    self.terminal_lines.append({
                        'text': self.current_command,
                        'style': 'green'
                    })
                    
                    # Add response
                    _, response = self.commands[self.current_command_index]
                    self.terminal_lines.append({
                        'text': response,
                        'style': 'dim white'
                    })
                    
                    # Progress bar
                    progress = int((self.current_command_index + 1) / len(self.commands) * 20)
                    self.terminal_lines.append({
                        'text': '[' + '█' * progress + '░' * (20 - progress) + '] ' + 
                               f"{(self.current_command_index + 1) * 20}%",
                        'style': 'cyan'
                    })
                    
                    self.current_command_index += 1
                    self.typing_progress = 0
                    self.current_command = ""
        else:
            # All commands complete
            if self.phase != "granted":
                self.phase = "granted"
                self.terminal_lines.append({
                    'text': "",
                    'style': None
                })
                self.terminal_lines.append({
                    'text': "ACCESS GRANTED",
                    'style': 'bold green'
                })
                self.terminal_lines.append({
                    'text': f"Welcome to {self.title}",
                    'style': 'bold white'
                })
        
        # Scroll if too many lines
        while len(self.terminal_lines) > self.height - 4:
            self.terminal_lines.pop(0)
    
        # Return the rendered content
        return self.render()
        
    def render(self):
        """Render hacker terminal."""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        style_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw terminal frame
        for x in range(self.width):
            grid[0][x] = '─'
            grid[self.height - 1][x] = '─'
            style_grid[0][x] = style_grid[self.height - 1][x] = 'green'
        
        for y in range(self.height):
            grid[y][0] = '│'
            grid[y][self.width - 1] = '│'
            style_grid[y][0] = style_grid[y][self.width - 1] = 'green'
        
        # Corners
        grid[0][0] = '┌'
        grid[0][self.width - 1] = '┐'
        grid[self.height - 1][0] = '└'
        grid[self.height - 1][self.width - 1] = '┘'
        
        # Terminal title
        title_text = " TERMINAL v2.0 "
        title_x = 2
        for i, char in enumerate(title_text):
            if title_x + i < self.width - 1:
                grid[0][title_x + i] = char
                style_grid[0][title_x + i] = 'bold green'
        
        # Draw terminal lines
        y_offset = 2
        for line in self.terminal_lines:
            if y_offset < self.height - 2:
                text = line['text']
                style = line['style']
                
                for i, char in enumerate(text):
                    if i + 2 < self.width - 2:
                        grid[y_offset][i + 2] = char
                        if style:
                            style_grid[y_offset][i + 2] = style
                
                y_offset += 1
        
        # Draw current command being typed
        if self.current_command and y_offset < self.height - 2:
            for i, char in enumerate(self.current_command):
                if i + 2 < self.width - 2:
                    grid[y_offset][i + 2] = char
                    style_grid[y_offset][i + 2] = 'green'
            
            # Blinking cursor
            cursor_x = len(self.current_command) + 2
            if cursor_x < self.width - 2 and int(self.phase_start_time * 2) % 2 == 0:
                grid[y_offset][cursor_x] = '█'
                style_grid[y_offset][cursor_x] = 'green'
        
        return self._grid_to_string(grid, style_grid)