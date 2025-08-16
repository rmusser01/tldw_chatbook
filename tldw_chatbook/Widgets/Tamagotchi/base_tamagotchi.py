"""
Base Tamagotchi Widget Implementation

Core widget class following Textual best practices.
"""

from typing import Optional, Dict, Any
from textual.reactive import reactive
from textual.widgets import Static
from textual.timer import Timer
from textual.app import RenderResult
from textual.events import Click
from textual.binding import Binding

from .tamagotchi_sprites import SpriteManager
from .tamagotchi_behaviors import BehaviorEngine
from .tamagotchi_storage import StorageAdapter, MemoryStorage
from .tamagotchi_messages import (
    TamagotchiInteraction,
    TamagotchiStateChange,
    TamagotchiDeath
)


class BaseTamagotchi(Static):
    """
    Core tamagotchi widget with state management.
    
    Follows Textual best practices:
    - Uses reactive properties for state management
    - Implements proper mount/unmount lifecycle
    - Provides DEFAULT_CSS for styling
    - Uses can_focus for interactivity
    - Handles events through Textual's event system
    """
    
    # Reactive properties for automatic UI updates
    happiness = reactive(50, layout=False)
    hunger = reactive(50, layout=False)
    energy = reactive(50, layout=False)
    health = reactive(100, layout=False)
    age = reactive(0.0, layout=False)
    
    # Visual state
    sprite = reactive("😊", layout=False)
    mood = reactive("happy", layout=False)
    is_sleeping = reactive(False, layout=False)
    is_sick = reactive(False, layout=False)
    
    # Make widget focusable for keyboard interaction
    can_focus = True
    
    # Keyboard bindings
    BINDINGS = [
        Binding("f", "feed", "Feed"),
        Binding("p", "play", "Play"),
        Binding("s", "sleep", "Sleep"),
        Binding("c", "clean", "Clean"),
        Binding("space", "pet", "Pet"),
    ]
    
    DEFAULT_CSS = """
    BaseTamagotchi {
        width: auto;
        height: 3;
        padding: 0 1;
        background: $surface;
        border: round $primary;
        content-align: center middle;
    }
    
    BaseTamagotchi:focus {
        border: round $accent;
    }
    
    BaseTamagotchi.sleeping {
        opacity: 0.7;
        border: round $secondary;
    }
    
    BaseTamagotchi.sick {
        border: round $error;
        background: $error 10%;
    }
    
    BaseTamagotchi.dead {
        opacity: 0.5;
        border: round $surface-lighten-2;
    }
    
    BaseTamagotchi.compact {
        height: 1;
        border: none;
        padding: 0;
    }
    
    BaseTamagotchi.minimal {
        width: 5;
        height: 1;
        border: none;
        padding: 0;
    }
    """
    
    def __init__(
        self,
        name: str = "Pet",
        personality: str = "balanced",
        update_interval: float = 30.0,
        storage: Optional[StorageAdapter] = None,
        sprite_theme: str = "emoji",
        size: str = "normal",
        *args,
        **kwargs
    ) -> None:
        """
        Initialize the tamagotchi widget.
        
        Args:
            name: Pet's name
            personality: Personality type affecting behavior
            update_interval: Seconds between automatic updates
            storage: Storage adapter for persistence
            sprite_theme: Visual theme for sprites
            size: Display size (normal, compact, minimal)
        """
        super().__init__(*args, **kwargs)
        
        self.pet_name = name
        self.personality_type = personality
        self.storage = storage or MemoryStorage()
        self.sprite_manager = SpriteManager(theme=sprite_theme)
        self.behavior_engine = BehaviorEngine(personality)
        self.size = size
        
        # Timer management
        self._update_timer: Optional[Timer] = None
        self._update_interval = update_interval
        self._animation_timer: Optional[Timer] = None
        
        # State tracking
        self._is_alive = True
        self._last_interaction_time = 0.0
        self._total_interactions = 0
        
        # Apply size class
        if size in ["compact", "minimal"]:
            self.add_class(size)
    
    def on_mount(self) -> None:
        """
        Called when widget is added to the app.
        Sets up timers and loads saved state.
        """
        # Load saved state
        self._load_state()
        
        # Start periodic updates if alive
        if self._is_alive:
            self._update_timer = self.set_interval(
                self._update_interval,
                self._periodic_update,
                name=f"tamagotchi-{self.id}-update"
            )
        
        # Initial sprite update
        self._update_sprite()
    
    def on_unmount(self) -> None:
        """
        Called when widget is removed from the app.
        Cleans up timers and saves state.
        """
        # Stop timers
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
        
        if self._animation_timer:
            self._animation_timer.stop()
            self._animation_timer = None
        
        # Save final state
        self._save_state()
    
    def _periodic_update(self) -> None:
        """
        Called periodically to update pet state.
        Applies time-based changes and checks conditions.
        """
        if not self._is_alive:
            return
        
        # Calculate time-based changes
        changes = self.behavior_engine.calculate_decay(self._update_interval)
        
        # Apply changes with bounds checking
        self.happiness = max(0, min(100, self.happiness + changes['happiness']))
        self.hunger = max(0, min(100, self.hunger + changes['hunger']))
        self.energy = max(0, min(100, self.energy + changes['energy']))
        
        # Update age (in hours)
        self.age += self._update_interval / 3600
        
        # Check critical conditions
        self._check_conditions()
        
        # Update mood and sprite
        self._update_mood()
        self._update_sprite()
        
        # Auto-save periodically
        self._save_state()
    
    def _check_conditions(self) -> None:
        """Check for state changes and critical conditions."""
        old_sleeping = self.is_sleeping
        old_sick = self.is_sick
        
        # Check if sleeping
        self.is_sleeping = self.energy < 20
        
        # Check if sick
        self.is_sick = self.health < 30 or self.hunger > 90
        
        # Check for death conditions
        if self.health <= 0 or self.happiness <= 0:
            self._handle_death("neglect")
        elif self.hunger >= 100:
            self._handle_death("starvation")
        
        # Update CSS classes based on state
        self.set_class(self.is_sleeping, "sleeping")
        self.set_class(self.is_sick, "sick")
        self.set_class(not self._is_alive, "dead")
        
        # Post state change messages
        if old_sleeping != self.is_sleeping:
            state = "sleeping" if self.is_sleeping else "awake"
            self.post_message(TamagotchiStateChange(self, "awake" if old_sleeping else "sleeping", state))
        
        if old_sick != self.is_sick:
            state = "sick" if self.is_sick else "healthy"
            self.post_message(TamagotchiStateChange(self, "healthy" if old_sick else "sick", state))
    
    def _update_mood(self) -> None:
        """Update mood based on current stats."""
        if not self._is_alive:
            self.mood = "dead"
        elif self.is_sick:
            self.mood = "sick"
        elif self.is_sleeping:
            self.mood = "sleepy"
        elif self.happiness > 75:
            self.mood = "happy"
        elif self.happiness > 50:
            self.mood = "neutral"
        elif self.happiness > 25:
            self.mood = "sad"
        else:
            self.mood = "very_sad"
        
        if self.hunger > 70:
            self.mood = "hungry"
    
    def _update_sprite(self) -> None:
        """Update visual sprite based on mood."""
        self.sprite = self.sprite_manager.get_sprite(self.mood)
    
    def _handle_death(self, cause: str) -> None:
        """Handle pet death."""
        if not self._is_alive:
            return
        
        self._is_alive = False
        self.mood = "dead"
        self._update_sprite()
        
        # Stop update timer
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer = None
        
        # Post death message
        self.post_message(TamagotchiDeath(self, cause, self.age))
        
        # Update classes
        self.add_class("dead")
    
    def on_click(self, event: Click) -> None:
        """
        Handle click events on the widget.
        Default action is to pet the tamagotchi.
        """
        if self._is_alive:
            self.interact("pet")
    
    def action_feed(self) -> None:
        """Action: Feed the pet."""
        self.interact("feed")
    
    def action_play(self) -> None:
        """Action: Play with the pet."""
        self.interact("play")
    
    def action_sleep(self) -> None:
        """Action: Put pet to sleep."""
        self.interact("sleep")
    
    def action_clean(self) -> None:
        """Action: Clean the pet."""
        self.interact("clean")
    
    def action_pet(self) -> None:
        """Action: Pet the tamagotchi."""
        self.interact("pet")
    
    def interact(self, action: str) -> None:
        """
        Process an interaction with the pet.
        
        Args:
            action: The interaction type (feed, play, pet, etc.)
        """
        if not self._is_alive:
            return
        
        # Get current stats
        current_stats = {
            'happiness': self.happiness,
            'hunger': self.hunger,
            'energy': self.energy,
            'health': self.health
        }
        
        # Process action through behavior engine
        response = self.behavior_engine.process_action(action, current_stats)
        
        # Apply stat changes
        for stat, change in response.get('changes', {}).items():
            if hasattr(self, stat):
                current = getattr(self, stat)
                setattr(self, stat, max(0, min(100, current + change)))
        
        # Update interaction counter
        self._total_interactions += 1
        
        # Trigger animation if specified
        if response.get('animation'):
            self._play_animation(response['animation'])
        
        # Post interaction message
        self.post_message(TamagotchiInteraction(self, action, response))
        
        # Update display immediately
        self._update_mood()
        self._update_sprite()
        self.refresh()
    
    def _play_animation(self, animation_type: str) -> None:
        """
        Play a simple animation.
        
        Args:
            animation_type: Type of animation to play
        """
        # Stop any existing animation
        if self._animation_timer:
            self._animation_timer.stop()
        
        # Get animation frames
        frames = self.sprite_manager.get_animation(animation_type)
        if not frames:
            return
        
        # Play animation frames
        frame_index = [0]
        
        def next_frame():
            if frame_index[0] < len(frames):
                self.sprite = frames[frame_index[0]]
                frame_index[0] += 1
                self.refresh()
            else:
                self._animation_timer.stop()
                self._animation_timer = None
                self._update_sprite()  # Restore normal sprite
        
        self._animation_timer = self.set_interval(0.2, next_frame)
    
    def render(self) -> RenderResult:
        """
        Render the widget content.
        
        Returns formatted string based on size setting.
        """
        if self.size == "minimal":
            # Minimal: Just the sprite
            return f"[{self.sprite}]"
        elif self.size == "compact":
            # Compact: Sprite and name
            return f"{self.sprite} {self.pet_name}"
        else:
            # Normal: Full stats display
            stats = f"❤️{int(self.happiness)} 🍽️{int(self.hunger)} ⚡{int(self.energy)}"
            return f"{self.sprite} {self.pet_name}\n{stats}\n{self.mood}"
    
    def _load_state(self) -> None:
        """Load saved state from storage."""
        try:
            state = self.storage.load(self.id or self.pet_name)
            if state:
                self.happiness = state.get('happiness', 50)
                self.hunger = state.get('hunger', 50)
                self.energy = state.get('energy', 50)
                self.health = state.get('health', 100)
                self.age = state.get('age', 0)
                self._is_alive = state.get('is_alive', True)
                self._total_interactions = state.get('total_interactions', 0)
        except Exception as e:
            self.log.error(f"Failed to load tamagotchi state: {e}")
    
    def _save_state(self) -> None:
        """Save current state to storage."""
        try:
            state = {
                'name': self.pet_name,
                'happiness': self.happiness,
                'hunger': self.hunger,
                'energy': self.energy,
                'health': self.health,
                'age': self.age,
                'personality': self.personality_type,
                'is_alive': self._is_alive,
                'total_interactions': self._total_interactions
            }
            self.storage.save(self.id or self.pet_name, state)
        except Exception as e:
            self.log.error(f"Failed to save tamagotchi state: {e}")
    
    def validate_happiness(self, value: float) -> float:
        """Validate happiness value stays in bounds."""
        return max(0, min(100, value))
    
    def validate_hunger(self, value: float) -> float:
        """Validate hunger value stays in bounds."""
        return max(0, min(100, value))
    
    def validate_energy(self, value: float) -> float:
        """Validate energy value stays in bounds."""
        return max(0, min(100, value))
    
    def validate_health(self, value: float) -> float:
        """Validate health value stays in bounds."""
        return max(0, min(100, value))


class CompactTamagotchi(BaseTamagotchi):
    """Compact version optimized for status bars."""
    
    def __init__(self, *args, **kwargs):
        kwargs['size'] = 'compact'
        super().__init__(*args, **kwargs)


class Tamagotchi(BaseTamagotchi):
    """Standard tamagotchi with full features."""
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('size', 'normal')
        super().__init__(*args, **kwargs)