"""WeatherSystem splash screen effect."""

import random
import time
from typing import Optional, Any, List, Tuple

from ..base_effect import BaseEffect, register_effect


@register_effect("weather_system")
class WeatherSystemEffect(BaseEffect):
    """Animated weather system with transitions."""
    
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
        
        # Weather states
        self.weather_cycle = ['sunny', 'cloudy', 'rainy', 'stormy', 'snowy', 'clearing']
        self.weather_duration = 1.5  # seconds per weather state
        
        # Weather elements
        self.sun = ["   \\│/   ", "  ─ ☀ ─  ", "   /│\\   "]
        self.cloud = ["   ☁☁☁   ", "  ☁☁☁☁☁  ", " ☁☁☁☁☁☁☁ "]
        self.rain_chars = ['·', ':', '│', '¦']
        self.snow_chars = ['·', '*', '❄', '✻']
        
        # Particle systems
        self.rain_particles = []
        self.snow_particles = []
        self.lightning_flash = 0
        
        # Initialize particles
        for _ in range(50):
            self.rain_particles.append({
                'x': random.randint(0, self.width),
                'y': random.randint(-10, 0),
                'speed': random.uniform(2, 4)
            })
        
        for _ in range(30):
            self.snow_particles.append({
                'x': random.randint(0, self.width),
                'y': random.randint(-10, 0),
                'speed': random.uniform(0.5, 1.5),
                'drift': random.uniform(-0.5, 0.5)
            })
    
    def update(self) -> Optional[str]:
        """Update the weather system animation."""
        elapsed = time.time() - self.start_time
        
        # Determine current weather
        weather_index = int(elapsed / self.weather_duration) % len(self.weather_cycle)
        current_weather = self.weather_cycle[weather_index]
        weather_progress = (elapsed % self.weather_duration) / self.weather_duration
        
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw title
        title = "TLDW CHATBOOK"
        subtitle = f"Weather: {current_weather.title()}"
        title_x = (self.width - len(title)) // 2
        subtitle_x = (self.width - len(subtitle)) // 2
        
        for i, char in enumerate(title):
            if 0 <= title_x + i < self.width:
                grid[2][title_x + i] = char
        
        for i, char in enumerate(subtitle):
            if 0 <= subtitle_x + i < self.width:
                grid[4][subtitle_x + i] = char
        
        # Draw weather elements based on current state
        if current_weather == 'sunny':
            # Draw sun
            sun_x = (self.width - len(self.sun[0])) // 2
            sun_y = 8
            for i, line in enumerate(self.sun):
                for j, char in enumerate(line):
                    if 0 <= sun_x + j < self.width and 0 <= sun_y + i < self.height:
                        if char != ' ':
                            grid[sun_y + i][sun_x + j] = char
        
        elif current_weather in ['cloudy', 'rainy', 'stormy', 'snowy']:
            # Draw clouds
            cloud_x = (self.width - len(self.cloud[0])) // 2
            cloud_y = 7
            for i, line in enumerate(self.cloud):
                for j, char in enumerate(line):
                    if 0 <= cloud_x + j < self.width and 0 <= cloud_y + i < self.height:
                        if char != ' ':
                            grid[cloud_y + i][cloud_x + j] = char
            
            if current_weather == 'rainy':
                # Update and draw rain
                for particle in self.rain_particles:
                    particle['y'] += particle['speed'] * self.speed
                    if particle['y'] > self.height:
                        particle['y'] = random.randint(-10, 0)
                        particle['x'] = random.randint(10, self.width - 10)
                    
                    y = int(particle['y'])
                    x = particle['x']
                    if 0 <= x < self.width and 0 <= y < self.height:
                        rain_char = self.rain_chars[int(y) % len(self.rain_chars)]
                        grid[y][x] = rain_char
            
            elif current_weather == 'stormy':
                # Lightning effect
                if random.random() < 0.05:  # 5% chance of lightning
                    self.lightning_flash = 3
                
                if self.lightning_flash > 0:
                    self.lightning_flash -= 1
                    # Draw lightning bolt
                    bolt_x = random.randint(20, self.width - 20)
                    bolt = ['╱', '╲', '╱']
                    for i, char in enumerate(bolt):
                        y = 10 + i * 2
                        x = bolt_x + i
                        if 0 <= x < self.width and 0 <= y < self.height:
                            grid[y][x] = char
                
                # Also show rain
                for particle in self.rain_particles[:30]:  # Fewer rain particles
                    particle['y'] += particle['speed'] * self.speed * 1.5
                    if particle['y'] > self.height:
                        particle['y'] = random.randint(-10, 0)
                        particle['x'] = random.randint(10, self.width - 10)
                    
                    y = int(particle['y'])
                    x = particle['x']
                    if 0 <= x < self.width and 0 <= y < self.height:
                        grid[y][x] = '│'
            
            elif current_weather == 'snowy':
                # Update and draw snow
                for particle in self.snow_particles:
                    particle['y'] += particle['speed'] * self.speed
                    particle['x'] += particle['drift'] * self.speed
                    
                    if particle['y'] > self.height:
                        particle['y'] = random.randint(-10, 0)
                        particle['x'] = random.randint(0, self.width)
                    
                    y = int(particle['y'])
                    x = int(particle['x'])
                    if 0 <= x < self.width and 0 <= y < self.height:
                        snow_char = self.snow_chars[random.randint(0, len(self.snow_chars) - 1)]
                        grid[y][x] = snow_char
        
        elif current_weather == 'clearing':
            # Transition effect - partial sun and clouds
            if weather_progress < 0.5:
                # Still some clouds
                cloud_x = int((self.width - len(self.cloud[0])) // 2 + weather_progress * 20)
                cloud_y = 7
                for i, line in enumerate(self.cloud):
                    for j, char in enumerate(line):
                        if 0 <= cloud_x + j < self.width and 0 <= cloud_y + i < self.height:
                            if char != ' ' and random.random() > weather_progress:
                                grid[cloud_y + i][cloud_x + j] = char
            
            # Sun appearing
            sun_x = int((self.width - len(self.sun[0])) // 2 - (1 - weather_progress) * 20)
            sun_y = 8
            for i, line in enumerate(self.sun):
                for j, char in enumerate(line):
                    if 0 <= sun_x + j < self.width and 0 <= sun_y + i < self.height:
                        if char != ' ' and random.random() < weather_progress:
                            grid[sun_y + i][sun_x + j] = char
        
        # Draw ground/horizon
        horizon_y = self.height - 5
        for x in range(self.width):
            grid[horizon_y][x] = '─'
        
        # Convert grid to string with styling
        lines = []
        for y, row in enumerate(grid):
            line = ""
            for x, char in enumerate(row):
                if y == 2 and char != ' ':  # Title
                    line += f"[bold white]{char}[/bold white]"
                elif y == 4 and char != ' ':  # Subtitle
                    line += f"[dim cyan]{char}[/dim cyan]"
                elif char == '☀':  # Sun
                    line += f"[bold yellow]{char}[/bold yellow]"
                elif char in '\\│/─':  # Sun rays or rain
                    if current_weather == 'sunny':
                        line += f"[yellow]{char}[/yellow]"
                    else:
                        line += f"[blue]{char}[/blue]"
                elif char == '☁':  # Clouds
                    if current_weather == 'stormy':
                        line += f"[dim white]{char}[/dim white]"
                    else:
                        line += f"[white]{char}[/white]"
                elif char in '·:¦':  # Rain
                    line += f"[blue]{char}[/blue]"
                elif char in '*❄✻':  # Snow
                    line += f"[bright_white]{char}[/bright_white]"
                elif char in '╱╲' and self.lightning_flash > 0:  # Lightning
                    line += f"[bold yellow]{char}[/bold yellow]"
                else:
                    line += char
            lines.append(line)
        
        return '\n'.join(lines)