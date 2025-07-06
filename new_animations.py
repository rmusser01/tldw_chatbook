
class FireworksEffect(BaseEffect):
    """Simulates fireworks explosions using ASCII characters."""
    
    @dataclass
    class Firework:
        x: float
        y: float
        vx: float  # Initial velocity x
        vy: float  # Initial velocity y
        age: float
        max_age: float
        exploded: bool
        color: str
        particles: List[Tuple[float, float, float, float]]  # x, y, vx, vy
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Celebrating innovation...",
        width: int = 80,
        height: int = 24,
        launch_rate: float = 0.5,  # Fireworks per second
        gravity: float = 15.0,
        explosion_chars: str = "*+x·°",
        trail_char: str = "|",
        colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan", "white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.launch_rate = launch_rate
        self.gravity = gravity
        self.explosion_chars = explosion_chars
        self.trail_char = trail_char
        self.colors = colors
        self.title_style = title_style
        
        self.fireworks: List[FireworksEffect.Firework] = []
        self.time_since_last_launch = 0.0
        self.last_update_time = time.time()
        self.revealed_chars = set()  # Track which title characters have been revealed
        
    def _launch_firework(self):
        """Launch a new firework from the bottom of the screen."""
        x = random.randint(10, self.display_width - 10)
        y = self.display_height - 1
        vx = random.uniform(-2, 2)
        vy = random.uniform(-20, -15)  # Negative for upward motion
        max_age = random.uniform(0.8, 1.5)
        color = random.choice(self.colors)
        
        self.fireworks.append(FireworksEffect.Firework(
            x=float(x), y=float(y), vx=vx, vy=vy, 
            age=0.0, max_age=max_age, exploded=False,
            color=color, particles=[]
        ))
    
    def _explode_firework(self, fw: 'FireworksEffect.Firework'):
        """Create explosion particles."""
        num_particles = random.randint(20, 40)
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(5, 15)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle) * 0.5  # Aspect ratio correction
            fw.particles.append((fw.x, fw.y, vx, vy))
        fw.exploded = True
        
        # Check if explosion reveals any title characters
        title_y = self.display_height // 2 - 2
        title_x = (self.display_width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            char_x = title_x + i
            char_y = title_y
            # If explosion is near this character, reveal it
            if abs(fw.x - char_x) < 10 and abs(fw.y - char_y) < 6:
                self.revealed_chars.add(i)
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Launch new fireworks
        self.time_since_last_launch += delta_time
        if self.time_since_last_launch >= 1.0 / self.launch_rate:
            self._launch_firework()
            self.time_since_last_launch = 0.0
        
        # Update existing fireworks
        active_fireworks = []
        for fw in self.fireworks:
            fw.age += delta_time
            
            if fw.age < fw.max_age * 2:  # Keep for a bit after explosion
                if not fw.exploded:
                    # Update position
                    fw.x += fw.vx * delta_time
                    fw.y += fw.vy * delta_time
                    fw.vy += self.gravity * delta_time  # Apply gravity
                    
                    # Check if it's time to explode
                    if fw.age >= fw.max_age or fw.vy > 0:
                        self._explode_firework(fw)
                
                # Update particles
                if fw.exploded:
                    new_particles = []
                    for px, py, pvx, pvy in fw.particles:
                        px += pvx * delta_time
                        py += pvy * delta_time
                        pvy += self.gravity * delta_time * 0.5  # Less gravity for particles
                        
                        if 0 <= px < self.display_width and 0 <= py < self.display_height:
                            new_particles.append((px, py, pvx, pvy))
                    fw.particles = new_particles
                
                if fw.exploded and len(fw.particles) > 0:
                    active_fireworks.append(fw)
                elif not fw.exploded:
                    active_fireworks.append(fw)
        
        self.fireworks = active_fireworks
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw fireworks
        for fw in self.fireworks:
            if not fw.exploded:
                # Draw trail
                x, y = int(fw.x), int(fw.y)
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    grid[y][x] = self.trail_char
                    styles[y][x] = fw.color
            else:
                # Draw explosion particles
                for px, py, _, _ in fw.particles:
                    x, y = int(px), int(py)
                    if 0 <= x < self.display_width and 0 <= y < self.display_height:
                        # Choose character based on particle age
                        char_index = int((fw.age - fw.max_age) / fw.max_age * len(self.explosion_chars))
                        char_index = max(0, min(char_index, len(self.explosion_chars) - 1))
                        
                        grid[y][x] = self.explosion_chars[char_index]
                        styles[y][x] = fw.color
        
        # Draw title (revealed characters only)
        if self.title:
            title_y = self.display_height // 2 - 2
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                if i in self.revealed_chars:
                    x = title_x + i
                    if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                        grid[title_y][x] = char
                        styles[title_y][x] = self.title_style
        
        # Always show subtitle
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class CircuitBoardEffect(BaseEffect):
    """Animated circuit board traces that light up and connect different components."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Connecting systems...",
        width: int = 80,
        height: int = 24,
        trace_speed: float = 20.0,  # Characters per second
        trace_chars: str = "─│┌┐└┘├┤┬┴┼",
        node_chars: str = "◆◇○●□■",
        active_color: str = "bright_green",
        inactive_color: str = "dim green",
        node_color: str = "yellow",
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.trace_speed = trace_speed
        self.trace_chars = trace_chars
        self.node_chars = node_chars
        self.active_color = active_color
        self.inactive_color = inactive_color
        self.node_color = node_color
        self.title_style = title_style
        
        self.grid = [[' ' for _ in range(width)] for _ in range(height)]
        self.active_cells = set()
        self.nodes = []
        self.traces = []
        self.current_trace_progress = 0.0
        self.last_update_time = time.time()
        
        self._generate_circuit()
        
    def _generate_circuit(self):
        """Generate a random circuit board layout."""
        # Place nodes
        num_nodes = random.randint(6, 10)
        for _ in range(num_nodes):
            x = random.randint(5, self.display_width - 5)
            y = random.randint(2, self.display_height - 3)
            node_char = random.choice(self.node_chars)
            self.nodes.append((x, y, node_char))
            self.grid[y][x] = node_char
        
        # Connect nodes with traces
        for i in range(len(self.nodes) - 1):
            start = self.nodes[i]
            end = self.nodes[i + 1]
            path = self._create_path(start[:2], end[:2])
            if path:
                self.traces.append(path)
    
    def _create_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a path between two points using Manhattan routing."""
        x1, y1 = start
        x2, y2 = end
        path = []
        
        # Simple L-shaped path
        if random.random() < 0.5:
            # Horizontal first
            for x in range(min(x1, x2), max(x1, x2) + 1):
                path.append((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                path.append((x2, y))
        else:
            # Vertical first
            for y in range(min(y1, y2), max(y1, y2) + 1):
                path.append((x1, y))
            for x in range(min(x1, x2), max(x1, x2) + 1):
                path.append((x, y2))
        
        return path
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update trace animation
        self.current_trace_progress += self.trace_speed * delta_time
        
        # Calculate which cells should be active
        total_cells = sum(len(trace) for trace in self.traces)
        active_count = int(self.current_trace_progress) % (total_cells + 20)  # Add pause
        
        self.active_cells.clear()
        cell_count = 0
        for trace in self.traces:
            for x, y in trace:
                if cell_count < active_count:
                    self.active_cells.add((x, y))
                cell_count += 1
        
        # Render
        display_grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw traces
        for trace_idx, trace in enumerate(self.traces):
            for i, (x, y) in enumerate(trace):
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    # Determine trace character based on connections
                    if i == 0 or i == len(trace) - 1:
                        continue  # Skip nodes
                    
                    prev_x, prev_y = trace[i-1] if i > 0 else (x, y)
                    next_x, next_y = trace[i+1] if i < len(trace)-1 else (x, y)
                    
                    # Choose appropriate character
                    if prev_x == next_x:  # Vertical
                        char = '│'
                    elif prev_y == next_y:  # Horizontal
                        char = '─'
                    elif (prev_x < x and next_y < y) or (prev_y < y and next_x < x):
                        char = '┌'
                    elif (prev_x > x and next_y < y) or (prev_y < y and next_x > x):
                        char = '┐'
                    elif (prev_x < x and next_y > y) or (prev_y > y and next_x < x):
                        char = '└'
                    else:
                        char = '┘'
                    
                    display_grid[y][x] = char
                    styles[y][x] = self.active_color if (x, y) in self.active_cells else self.inactive_color
        
        # Draw nodes
        for x, y, char in self.nodes:
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                display_grid[y][x] = char
                styles[y][x] = self.node_color
        
        # Draw title
        if self.title:
            title_y = 1
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    display_grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = self.display_height - 2
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
            for i, char in enumerate(self.subtitle):
                x = subtitle_x + i
                if 0 <= x < self.display_width and 0 <= subtitle_y < self.display_height:
                    display_grid[subtitle_y][x] = char
                    styles[subtitle_y][x] = "white"
        
        # Convert to Rich markup
        output_lines = []
        for y in range(self.display_height):
            line_segments = []
            for x in range(self.display_width):
                char = display_grid[y][x]
                style = styles[y][x]
                
                if style:
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class PixelDissolveEffect(BaseEffect):
    """The screen starts filled with random ASCII characters that gradually dissolve away."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Revealing clarity...",
        width: int = 80,
        height: int = 24,
        dissolve_rate: float = 0.02,  # Percentage per frame
        noise_chars: str = "█▓▒░╳╱╲┃━┏┓┗┛",
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.dissolve_rate = dissolve_rate
        self.noise_chars = noise_chars
        self.title_style = title_style
        
        # Initialize with all pixels as noise
        self.dissolved_pixels = set()
        self.total_pixels = width * height
        
    def update(self) -> Optional[str]:
        # Calculate how many pixels to dissolve this frame
        current_dissolved = len(self.dissolved_pixels)
        target_dissolved = min(self.total_pixels, 
                              current_dissolved + int(self.total_pixels * self.dissolve_rate))
        
        # Dissolve random pixels
        while len(self.dissolved_pixels) < target_dissolved:
            x = random.randint(0, self.display_width - 1)
            y = random.randint(0, self.display_height - 1)
            self.dissolved_pixels.add((x, y))
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Fill with noise or clear based on dissolved state
        for y in range(self.display_height):
            for x in range(self.display_width):
                if (x, y) not in self.dissolved_pixels:
                    grid[y][x] = random.choice(self.noise_chars)
                    styles[y][x] = random.choice(["dim white", "dim gray", "dim black"])
        
        # Always show title and subtitle (on top of noise)
        if self.title:
            title_y = self.display_height // 2 - 2
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class TetrisBlockEffect(BaseEffect):
    """Tetris-style blocks fall from the top and stack up to form the title text."""
    
    @dataclass
    class Block:
        x: int
        y: float
        char: str
        target_y: int
        color: str
        falling: bool = True
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Building blocks...",
        width: int = 80,
        height: int = 24,
        fall_speed: float = 8.0,  # Blocks per second
        block_chars: str = "█",
        colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.fall_speed = fall_speed
        self.block_chars = block_chars
        self.colors = colors
        self.title_style = title_style
        
        self.blocks: List[TetrisBlockEffect.Block] = []
        self.last_update_time = time.time()
        self.spawn_delay = 0.1
        self.time_since_spawn = 0.0
        self.title_positions = []
        
        self._calculate_title_positions()
        self.spawn_index = 0
        
    def _calculate_title_positions(self):
        """Calculate where each character of the title should be."""
        title_y = self.display_height // 2 - 2
        title_x = (self.display_width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            if char != ' ':
                self.title_positions.append((title_x + i, title_y, char))
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Spawn new blocks
        self.time_since_spawn += delta_time
        if self.time_since_spawn >= self.spawn_delay and self.spawn_index < len(self.title_positions):
            x, y, char = self.title_positions[self.spawn_index]
            color = random.choice(self.colors)
            self.blocks.append(TetrisBlockEffect.Block(
                x=x, y=0, char=char, target_y=y, color=color
            ))
            self.spawn_index += 1
            self.time_since_spawn = 0.0
        
        # Update falling blocks
        for block in self.blocks:
            if block.falling:
                block.y += self.fall_speed * delta_time
                if block.y >= block.target_y:
                    block.y = block.target_y
                    block.falling = False
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw blocks
        for block in self.blocks:
            y = int(block.y)
            if 0 <= block.x < self.display_width and 0 <= y < self.display_height:
                grid[y][block.x] = block.char
                styles[y][block.x] = block.color if block.falling else self.title_style
        
        # Always show subtitle
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 1
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class SpiralGalaxyEffect(BaseEffect):
    """Creates a rotating spiral galaxy pattern with ASCII stars and cosmic dust."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Exploring the cosmos...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.2,  # Radians per second
        num_arms: int = 3,
        star_chars: str = "·∙*✦✧★☆",
        star_colors: List[str] = ["white", "bright_white", "yellow", "cyan", "dim white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.num_arms = num_arms
        self.star_chars = star_chars
        self.star_colors = star_colors
        self.title_style = title_style
        
        self.rotation = 0.0
        self.last_update_time = time.time()
        self.stars = []
        
        self._generate_stars()
        
    def _generate_stars(self):
        """Generate stars in spiral pattern."""
        center_x = self.display_width / 2
        center_y = self.display_height / 2
        
        for i in range(200):  # Number of stars
            # Spiral parameters
            angle = random.uniform(0, 4 * math.pi)
            radius = random.uniform(0, min(center_x, center_y) - 5)
            
            # Add arm structure
            arm = random.randint(0, self.num_arms - 1)
            arm_angle = (2 * math.pi * arm) / self.num_arms
            
            # Logarithmic spiral
            spiral_angle = angle + arm_angle + (radius * 0.1)
            
            x = center_x + radius * math.cos(spiral_angle)
            y = center_y + radius * math.sin(spiral_angle) * 0.5  # Aspect ratio
            
            char = random.choice(self.star_chars)
            color = random.choice(self.star_colors)
            
            self.stars.append({
                'radius': radius,
                'angle': spiral_angle,
                'char': char,
                'color': color,
                'brightness': random.uniform(0.3, 1.0)
            })
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.rotation += self.rotation_speed * delta_time
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width / 2
        center_y = self.display_height / 2
        
        # Draw stars
        for star in self.stars:
            # Rotate star
            angle = star['angle'] + self.rotation
            x = int(center_x + star['radius'] * math.cos(angle))
            y = int(center_y + star['radius'] * math.sin(angle) * 0.5)
            
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                # Twinkle effect
                if random.random() < star['brightness']:
                    grid[y][x] = star['char']
                    styles[y][x] = star['color']
        
        # Draw title emerging from galactic center
        if self.title:
            title_y = int(center_y)
            title_x = int((self.display_width - len(self.title)) / 2)
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = int(center_y) + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class MorphingShapeEffect(BaseEffect):
    """Geometric ASCII shapes that morph and transform into one another."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Transforming reality...",
        width: int = 80,
        height: int = 24,
        morph_speed: float = 0.5,  # Morphs per second
        shape_chars: str = "○□△▽◇◆",
        shape_colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.morph_speed = morph_speed
        self.shape_chars = shape_chars
        self.shape_colors = shape_colors
        self.title_style = title_style
        
        self.morph_progress = 0.0
        self.current_shape = 0
        self.shapes = ['circle', 'square', 'triangle', 'diamond']
        self.last_update_time = time.time()
        
    def _draw_circle(self, center_x: int, center_y: int, radius: int, progress: float):
        """Draw a circle shape."""
        points = []
        num_points = int(50 * progress)
        for i in range(num_points):
            angle = (2 * math.pi * i) / 50
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle) * 0.5)
            points.append((x, y))
        return points
    
    def _draw_square(self, center_x: int, center_y: int, size: int, progress: float):
        """Draw a square shape."""
        points = []
        half_size = size // 2
        perimeter = size * 4
        points_to_draw = int(perimeter * progress)
        
        for i in range(points_to_draw):
            if i < size:  # Top edge
                points.append((center_x - half_size + i, center_y - half_size))
            elif i < size * 2:  # Right edge
                points.append((center_x + half_size, center_y - half_size + (i - size)))
            elif i < size * 3:  # Bottom edge
                points.append((center_x + half_size - (i - size * 2), center_y + half_size))
            else:  # Left edge
                points.append((center_x - half_size, center_y + half_size - (i - size * 3)))
        
        return points
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.morph_progress += self.morph_speed * delta_time
        
        if self.morph_progress >= 1.0:
            self.morph_progress = 0.0
            self.current_shape = (self.current_shape + 1) % len(self.shapes)
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        
        # Draw current shape
        shape_type = self.shapes[self.current_shape]
        if shape_type == 'circle':
            points = self._draw_circle(center_x, center_y, 10, self.morph_progress)
        elif shape_type == 'square':
            points = self._draw_square(center_x, center_y, 20, self.morph_progress)
        else:
            points = []
        
        # Draw points
        for x, y in points:
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                char_index = int(self.morph_progress * (len(self.shape_chars) - 1))
                grid[y][x] = self.shape_chars[char_index]
                color_index = self.current_shape % len(self.shape_colors)
                styles[y][x] = self.shape_colors[color_index]
        
        # Draw title (always visible)
        if self.title:
            title_y = center_y
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = center_y + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class ParticleSwarmEffect(BaseEffect):
    """A swarm of ASCII characters that move with flocking behavior."""
    
    @dataclass
    class Particle:
        x: float
        y: float
        vx: float
        vy: float
        char: str
        color: str
        target_index: Optional[int] = None
        
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Swarming intelligence...",
        width: int = 80,
        height: int = 24,
        num_particles: int = 50,
        swarm_speed: float = 10.0,
        cohesion: float = 0.01,
        separation: float = 0.1,
        alignment: float = 0.05,
        particle_chars: str = "·∙○◦",
        particle_colors: List[str] = ["cyan", "blue", "white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.num_particles = num_particles
        self.swarm_speed = swarm_speed
        self.cohesion = cohesion
        self.separation = separation
        self.alignment = alignment
        self.particle_chars = particle_chars
        self.particle_colors = particle_colors
        self.title_style = title_style
        
        self.particles: List[ParticleSwarmEffect.Particle] = []
        self.last_update_time = time.time()
        self.formation_mode = False
        self.formation_progress = 0.0
        
        self._init_particles()
        self._calculate_title_positions()
        
    def _init_particles(self):
        """Initialize particles with random positions and velocities."""
        for i in range(self.num_particles):
            self.particles.append(ParticleSwarmEffect.Particle(
                x=random.uniform(0, self.display_width),
                y=random.uniform(0, self.display_height),
                vx=random.uniform(-2, 2),
                vy=random.uniform(-2, 2),
                char=random.choice(self.particle_chars),
                color=random.choice(self.particle_colors)
            ))
    
    def _calculate_title_positions(self):
        """Calculate target positions for title formation."""
        self.title_positions = []
        title_y = self.display_height // 2
        title_x = (self.display_width - len(self.title)) // 2
        
        for i, char in enumerate(self.title):
            if char != ' ':
                self.title_positions.append((title_x + i, title_y))
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Switch to formation mode after some time
        elapsed = current_time - self.start_time
        if elapsed > 3.0 and not self.formation_mode:
            self.formation_mode = True
            # Assign particles to title positions
            for i, particle in enumerate(self.particles[:len(self.title_positions)]):
                particle.target_index = i
        
        # Update particles
        for particle in self.particles:
            if self.formation_mode and particle.target_index is not None:
                # Move towards target position
                target_x, target_y = self.title_positions[particle.target_index]
                dx = target_x - particle.x
                dy = target_y - particle.y
                
                particle.vx = dx * 0.1
                particle.vy = dy * 0.1
            else:
                # Flocking behavior
                # Find nearby particles
                neighbors = []
                for other in self.particles:
                    if other != particle:
                        dist = math.sqrt((other.x - particle.x)**2 + (other.y - particle.y)**2)
                        if dist < 10:
                            neighbors.append(other)
                
                if neighbors:
                    # Cohesion
                    avg_x = sum(n.x for n in neighbors) / len(neighbors)
                    avg_y = sum(n.y for n in neighbors) / len(neighbors)
                    particle.vx += (avg_x - particle.x) * self.cohesion
                    particle.vy += (avg_y - particle.y) * self.cohesion
                    
                    # Separation
                    for neighbor in neighbors:
                        dist = math.sqrt((neighbor.x - particle.x)**2 + (neighbor.y - particle.y)**2)
                        if dist < 5 and dist > 0:
                            particle.vx -= (neighbor.x - particle.x) / dist * self.separation
                            particle.vy -= (neighbor.y - particle.y) / dist * self.separation
                    
                    # Alignment
                    avg_vx = sum(n.vx for n in neighbors) / len(neighbors)
                    avg_vy = sum(n.vy for n in neighbors) / len(neighbors)
                    particle.vx += (avg_vx - particle.vx) * self.alignment
            
            # Limit speed
            speed = math.sqrt(particle.vx**2 + particle.vy**2)
            if speed > self.swarm_speed:
                particle.vx = (particle.vx / speed) * self.swarm_speed
                particle.vy = (particle.vy / speed) * self.swarm_speed
            
            # Update position
            particle.x += particle.vx * delta_time
            particle.y += particle.vy * delta_time
            
            # Wrap around edges
            particle.x = particle.x % self.display_width
            particle.y = particle.y % self.display_height
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        # Draw particles
        for particle in self.particles:
            x, y = int(particle.x), int(particle.y)
            if 0 <= x < self.display_width and 0 <= y < self.display_height:
                if self.formation_mode and particle.target_index is not None and particle.target_index < len(self.title):
                    grid[y][x] = self.title[particle.target_index]
                    styles[y][x] = self.title_style
                else:
                    grid[y][x] = particle.char
                    styles[y][x] = particle.color
        
        # Always show subtitle
        if self.subtitle:
            subtitle_y = self.display_height // 2 + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)


class ASCIIKaleidoscopeEffect(BaseEffect):
    """Creates symmetrical, rotating kaleidoscope patterns using ASCII characters."""
    
    def __init__(
        self,
        parent_widget: Any,
        title: str = "tldw chatbook",
        subtitle: str = "Infinite patterns...",
        width: int = 80,
        height: int = 24,
        rotation_speed: float = 0.3,
        num_mirrors: int = 6,
        pattern_chars: str = "◆◇○●□■▲▼",
        pattern_colors: List[str] = ["red", "yellow", "blue", "green", "magenta", "cyan", "white"],
        title_style: str = "bold white",
        **kwargs
    ):
        super().__init__(parent_widget, **kwargs)
        self.title = title
        self.subtitle = subtitle
        self.display_width = width
        self.display_height = height
        self.rotation_speed = rotation_speed
        self.num_mirrors = num_mirrors
        self.pattern_chars = pattern_chars
        self.pattern_colors = pattern_colors
        self.title_style = title_style
        
        self.rotation = 0.0
        self.last_update_time = time.time()
        self.pattern_elements = []
        
        self._generate_pattern()
        
    def _generate_pattern(self):
        """Generate random pattern elements in one segment."""
        segment_angle = (2 * math.pi) / self.num_mirrors
        
        for _ in range(10):  # Number of elements per segment
            angle = random.uniform(0, segment_angle)
            radius = random.uniform(3, 15)
            char = random.choice(self.pattern_chars)
            color = random.choice(self.pattern_colors)
            
            self.pattern_elements.append({
                'angle': angle,
                'radius': radius,
                'char': char,
                'color': color
            })
    
    def update(self) -> Optional[str]:
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time
        
        self.rotation += self.rotation_speed * delta_time
        
        # Render
        grid = [[' ' for _ in range(self.display_width)] for _ in range(self.display_height)]
        styles = [[None for _ in range(self.display_width)] for _ in range(self.display_height)]
        
        center_x = self.display_width // 2
        center_y = self.display_height // 2
        
        # Draw kaleidoscope pattern
        for element in self.pattern_elements:
            for mirror in range(self.num_mirrors):
                # Calculate mirrored angle
                mirror_angle = (2 * math.pi * mirror) / self.num_mirrors
                angle = element['angle'] + mirror_angle + self.rotation
                
                # Calculate position
                x = int(center_x + element['radius'] * math.cos(angle))
                y = int(center_y + element['radius'] * math.sin(angle) * 0.5)
                
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    grid[y][x] = element['char']
                    styles[y][x] = element['color']
                
                # Add reflection
                if mirror % 2 == 0:
                    angle_reflected = -element['angle'] + mirror_angle + self.rotation
                    x = int(center_x + element['radius'] * math.cos(angle_reflected))
                    y = int(center_y + element['radius'] * math.sin(angle_reflected) * 0.5)
                    
                    if 0 <= x < self.display_width and 0 <= y < self.display_height:
                        grid[y][x] = element['char']
                        styles[y][x] = element['color']
        
        # Draw title in center
        if self.title:
            title_y = center_y
            title_x = (self.display_width - len(self.title)) // 2
            
            for i, char in enumerate(self.title):
                x = title_x + i
                if 0 <= x < self.display_width and 0 <= title_y < self.display_height:
                    grid[title_y][x] = char
                    styles[title_y][x] = self.title_style
        
        # Draw subtitle
        if self.subtitle:
            subtitle_y = center_y + 3
            subtitle_x = (self.display_width - len(self.subtitle)) // 2
            
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
                    line_segments.append(f"[{style}]{char.replace('[', r'\[')}[/{style}]")
                else:
                    line_segments.append(char)
            
            output_lines.append(''.join(line_segments))
        
        return '\n'.join(output_lines)