# Textual Tamagotchis Module

A modular, customizable tamagotchi widget system for Textual applications. This document provides comprehensive guidance for implementing, customizing, and integrating virtual pets into any Textual-based TUI application.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Customization Guide](#customization-guide)
5. [Integration Examples](#integration-examples)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Usage

```python
from textual.app import App, ComposeResult
from tldw_chatbook.Widgets.Tamagotchi import Tamagotchi

class MyApp(App):
    def compose(self) -> ComposeResult:
        # Add a simple tamagotchi to your app
        yield Tamagotchi(
            name="Pixel",
            id="my-pet"
        )

if __name__ == "__main__":
    MyApp().run()
```

### Status Bar Integration

```python
from tldw_chatbook.Widgets.Tamagotchi import CompactTamagotchi

class FooterWithPet(Widget):
    def compose(self) -> ComposeResult:
        yield Static("Status: Ready")
        yield CompactTamagotchi(name="Bit", size="minimal")
        yield Static("CPU: 42%")
```

## Architecture Overview

### Design Principles

1. **Modularity**: Each component is independent and replaceable
2. **Extensibility**: Easy to add new behaviors, sprites, and features
3. **Performance**: Efficient rendering using Line API for frequent updates
4. **Integration**: Drop-in widget that works with existing Textual apps
5. **Customization**: Theming, behaviors, and storage are all configurable

### Component Hierarchy

```
BaseTamagotchi (Core Widget)
├── SpriteManager (Visual representation)
├── BehaviorEngine (Personality & actions)
├── StateManager (Stats & conditions)
├── StorageAdapter (Persistence)
└── MessageBus (Event communication)
```

## Core Components

### 1. Base Tamagotchi Widget

```python
from textual.reactive import reactive
from textual.widgets import Static
from textual.timer import Timer

class BaseTamagotchi(Static):
    """Core tamagotchi widget with state management"""
    
    # Reactive properties for automatic UI updates
    happiness = reactive(50, layout=False)
    hunger = reactive(50, layout=False)
    energy = reactive(50, layout=False)
    health = reactive(100, layout=False)
    age = reactive(0, layout=False)
    
    # Visual state
    sprite = reactive("😊", layout=False)
    mood = reactive("happy", layout=False)
    
    DEFAULT_CSS = """
    BaseTamagotchi {
        width: auto;
        height: 3;
        padding: 0 1;
        background: $surface;
        border: round $primary;
    }
    
    BaseTamagotchi.sleeping {
        opacity: 0.7;
        border: round $secondary;
    }
    
    BaseTamagotchi.sick {
        border: round $error;
        background: $error 10%;
    }
    
    BaseTamagotchi.compact {
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.personality = personality
        self.storage = storage or MemoryStorage()
        self.sprite_manager = SpriteManager(theme=sprite_theme)
        self.behavior_engine = BehaviorEngine(personality)
        self.size = size
        
        # Set up automatic updates
        self._update_timer: Optional[Timer] = None
        self._update_interval = update_interval
        
    def on_mount(self) -> None:
        """Initialize timers and load state when widget is mounted"""
        self._load_state()
        self._update_timer = self.set_interval(
            self._update_interval,
            self._periodic_update,
            name=f"tamagotchi-update-{self.id}"
        )
        
    def on_unmount(self) -> None:
        """Clean up and save state when widget is unmounted"""
        if self._update_timer:
            self._update_timer.stop()
        self._save_state()
    
    def _periodic_update(self) -> None:
        """Called periodically to update pet state"""
        # Apply time-based stat changes
        changes = self.behavior_engine.calculate_decay(self._update_interval)
        
        self.happiness = max(0, min(100, self.happiness + changes['happiness']))
        self.hunger = max(0, min(100, self.hunger + changes['hunger']))
        self.energy = max(0, min(100, self.energy + changes['energy']))
        
        # Update age
        self.age += self._update_interval / 3600  # Age in hours
        
        # Check for state changes
        self._check_conditions()
        self._update_mood()
        self._save_state()
    
    def on_click(self) -> None:
        """Handle click interactions"""
        self.interact("pet")
    
    def interact(self, action: str) -> None:
        """Process interactions with the pet"""
        response = self.behavior_engine.process_action(
            action,
            {
                'happiness': self.happiness,
                'hunger': self.hunger,
                'energy': self.energy,
                'health': self.health
            }
        )
        
        # Apply stat changes
        for stat, change in response['changes'].items():
            setattr(self, stat, max(0, min(100, getattr(self, stat) + change)))
        
        # Trigger animation
        if response.get('animation'):
            self._play_animation(response['animation'])
        
        # Post message for parent widgets
        self.post_message(
            TamagotchiInteraction(self, action, response)
        )
    
    def render(self) -> str:
        """Render the tamagotchi display"""
        if self.size == "minimal":
            return f"[{self.sprite}]"
        elif self.size == "compact":
            return f"{self.sprite} {self.name}"
        else:
            stats = f"H:{self.happiness} F:{self.hunger} E:{self.energy}"
            return f"{self.sprite} {self.name}\n{stats}\n[{self.mood}]"
```

### 2. Sprite System

```python
class SpriteManager:
    """Manages visual representations of the tamagotchi"""
    
    EMOJI_SPRITES = {
        'happy': ['😊', '😄', '🥰'],
        'neutral': ['😐', '🙂', '😑'],
        'sad': ['😢', '😭', '😞'],
        'hungry': ['😋', '🤤', '😫'],
        'sleepy': ['😴', '😪', '🥱'],
        'sick': ['🤢', '🤒', '😷'],
        'dead': ['💀', '👻', '⚰️'],
        'baby': ['🥚', '🐣', '🐥'],
        'teen': ['🐤', '🐦', '🦆'],
        'adult': ['🐓', '🦅', '🦜']
    }
    
    ASCII_SPRITES = {
        'happy': [
            r"^_^",
            r"^o^",
            r"(◕‿◕)"
        ],
        'neutral': [
            r"-_-",
            r"o_o",
            r"(._.|"
        ],
        'sad': [
            r"T_T",
            r";_;",
            r"(╥﹏╥)"
        ],
        'hungry': [
            r"@_@",
            r"*o*",
            r"(｡◕‿◕｡)"
        ],
        'sleepy': [
            r"u_u",
            r"-.-",
            r"(－ω－) zzZ"
        ],
        'sick': [
            r"x_x",
            r"+_+",
            r"(×﹏×)"
        ],
        'dead': [
            r"X_X",
            r"✝_✝",
            r"(✖╭╮✖)"
        ]
    }
    
    def __init__(self, theme: str = "emoji"):
        self.theme = theme
        self.custom_sprites = {}
        self.animation_frames = {}
        
    def get_sprite(self, mood: str, variation: int = 0) -> str:
        """Get sprite for current mood"""
        sprite_set = self.EMOJI_SPRITES if self.theme == "emoji" else self.ASCII_SPRITES
        
        if mood in self.custom_sprites:
            sprites = self.custom_sprites[mood]
        elif mood in sprite_set:
            sprites = sprite_set[mood]
        else:
            sprites = sprite_set.get('neutral', ['?'])
            
        return sprites[variation % len(sprites)]
    
    def register_sprite(self, mood: str, sprites: List[str]) -> None:
        """Register custom sprites for a mood"""
        self.custom_sprites[mood] = sprites
    
    def get_animation(self, action: str) -> List[str]:
        """Get animation frames for an action"""
        if action == "bounce":
            return ["⤴", "⤵", "⤴", "⤵"]
        elif action == "spin":
            return ["◐", "◓", "◑", "◒"]
        elif action == "heart":
            return ["♡", "💕", "💖", "💕", "♡"]
        return []
```

### 3. Behavior Engine

```python
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class Personality:
    """Defines personality traits affecting behavior"""
    name: str
    happiness_decay: float  # Points per minute
    hunger_increase: float  # Points per minute  
    energy_decay: float     # Points per minute
    social_need: float      # Multiplier for interaction effects
    resilience: float       # Resistance to negative effects
    
PERSONALITIES = {
    'balanced': Personality(
        name='balanced',
        happiness_decay=-0.5,
        hunger_increase=1.0,
        energy_decay=-0.3,
        social_need=1.0,
        resilience=1.0
    ),
    'energetic': Personality(
        name='energetic',
        happiness_decay=-0.3,
        hunger_increase=1.5,
        energy_decay=-0.6,
        social_need=1.2,
        resilience=0.8
    ),
    'lazy': Personality(
        name='lazy',
        happiness_decay=-0.2,
        hunger_increase=0.8,
        energy_decay=-0.1,
        social_need=0.7,
        resilience=1.2
    ),
    'needy': Personality(
        name='needy',
        happiness_decay=-1.0,
        hunger_increase=1.2,
        energy_decay=-0.4,
        social_need=2.0,
        resilience=0.6
    )
}

class BehaviorEngine:
    """Manages pet behavior and personality"""
    
    def __init__(self, personality_type: str = "balanced"):
        self.personality = PERSONALITIES.get(
            personality_type, 
            PERSONALITIES['balanced']
        )
        self.action_effects = self._init_action_effects()
        
    def _init_action_effects(self) -> Dict[str, Dict[str, Any]]:
        """Initialize action effect mappings"""
        return {
            'feed': {
                'changes': {'hunger': -30, 'happiness': 5, 'energy': 10},
                'animation': 'eating',
                'cooldown': 60
            },
            'play': {
                'changes': {'happiness': 20, 'energy': -15, 'hunger': 5},
                'animation': 'bounce',
                'cooldown': 30
            },
            'pet': {
                'changes': {'happiness': 10 * self.personality.social_need},
                'animation': 'heart',
                'cooldown': 10
            },
            'sleep': {
                'changes': {'energy': 50, 'happiness': 5},
                'animation': 'sleeping',
                'cooldown': 300
            },
            'medicine': {
                'changes': {'health': 30, 'happiness': -10},
                'animation': 'healing',
                'cooldown': 120
            },
            'clean': {
                'changes': {'health': 10, 'happiness': 15},
                'animation': 'sparkle',
                'cooldown': 60
            }
        }
    
    def calculate_decay(self, time_delta: float) -> Dict[str, float]:
        """Calculate stat changes over time"""
        minutes = time_delta / 60
        return {
            'happiness': self.personality.happiness_decay * minutes,
            'hunger': self.personality.hunger_increase * minutes,
            'energy': self.personality.energy_decay * minutes
        }
    
    def process_action(
        self, 
        action: str, 
        current_stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Process an action and return effects"""
        if action not in self.action_effects:
            return {'changes': {}, 'success': False}
        
        effect = self.action_effects[action].copy()
        
        # Modify effects based on current state
        if current_stats['health'] < 30:
            # Sick pets respond poorly to most actions
            if action != 'medicine':
                effect['changes']['happiness'] *= 0.5
        
        if current_stats['energy'] < 20:
            # Tired pets need sleep
            if action == 'play':
                effect['changes']['happiness'] *= 0.3
                effect['changes']['energy'] *= 1.5
        
        # Apply personality modifiers
        effect['changes']['happiness'] = effect['changes'].get('happiness', 0) * self.personality.resilience
        
        return effect
```

### 4. Storage System

```python
from abc import ABC, abstractmethod
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional

class StorageAdapter(ABC):
    """Abstract base for storage implementations"""
    
    @abstractmethod
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """Load pet state"""
        pass
    
    @abstractmethod
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """Save pet state"""
        pass
    
    @abstractmethod
    def delete(self, pet_id: str) -> bool:
        """Delete pet data"""
        pass

class JSONStorage(StorageAdapter):
    """JSON file storage implementation"""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath).expanduser()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """Load pet state from JSON file"""
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    return data.get(pet_id)
        except Exception as e:
            print(f"Error loading pet data: {e}")
        return None
    
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """Save pet state to JSON file"""
        try:
            data = {}
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
            
            data[pet_id] = state
            
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving pet data: {e}")
            return False
    
    def delete(self, pet_id: str) -> bool:
        """Delete pet from JSON file"""
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                
                if pet_id in data:
                    del data[pet_id]
                    with open(self.filepath, 'w') as f:
                        json.dump(data, f, indent=2)
                    return True
        except Exception as e:
            print(f"Error deleting pet data: {e}")
        return False

class SQLiteStorage(StorageAdapter):
    """SQLite database storage implementation"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tamagotchis (
                    pet_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    happiness REAL DEFAULT 50,
                    hunger REAL DEFAULT 50,
                    energy REAL DEFAULT 50,
                    health REAL DEFAULT 100,
                    age REAL DEFAULT 0,
                    personality TEXT DEFAULT 'balanced',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """Load pet state from database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM tamagotchis WHERE pet_id = ?",
                (pet_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None
    
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """Save pet state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO tamagotchis 
                    (pet_id, name, happiness, hunger, energy, health, age, personality, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    pet_id,
                    state.get('name', 'Pet'),
                    state.get('happiness', 50),
                    state.get('hunger', 50),
                    state.get('energy', 50),
                    state.get('health', 100),
                    state.get('age', 0),
                    state.get('personality', 'balanced')
                ))
            return True
        except Exception as e:
            print(f"Error saving to database: {e}")
            return False
    
    def delete(self, pet_id: str) -> bool:
        """Delete pet from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM tamagotchis WHERE pet_id = ?", (pet_id,))
            return True
        except Exception as e:
            print(f"Error deleting from database: {e}")
            return False

class MemoryStorage(StorageAdapter):
    """In-memory storage for testing"""
    
    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}
    
    def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(pet_id)
    
    def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        self.data[pet_id] = state
        return True
    
    def delete(self, pet_id: str) -> bool:
        if pet_id in self.data:
            del self.data[pet_id]
            return True
        return False
```

### 5. Message System

```python
from textual.message import Message
from typing import Any, Dict

class TamagotchiMessage(Message):
    """Base message for tamagotchi events"""
    
    def __init__(self, tamagotchi: 'BaseTamagotchi'):
        super().__init__()
        self.tamagotchi = tamagotchi
        self.pet_id = tamagotchi.id
        self.pet_name = tamagotchi.name

class TamagotchiInteraction(TamagotchiMessage):
    """Sent when user interacts with tamagotchi"""
    
    def __init__(
        self, 
        tamagotchi: 'BaseTamagotchi',
        action: str,
        result: Dict[str, Any]
    ):
        super().__init__(tamagotchi)
        self.action = action
        self.result = result

class TamagotchiStateChange(TamagotchiMessage):
    """Sent when tamagotchi state changes significantly"""
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        old_state: str,
        new_state: str
    ):
        super().__init__(tamagotchi)
        self.old_state = old_state
        self.new_state = new_state

class TamagotchiEvolution(TamagotchiMessage):
    """Sent when tamagotchi evolves"""
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        from_stage: str,
        to_stage: str
    ):
        super().__init__(tamagotchi)
        self.from_stage = from_stage
        self.to_stage = to_stage

class TamagotchiAchievement(TamagotchiMessage):
    """Sent when achievement is unlocked"""
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        achievement: str,
        description: str
    ):
        super().__init__(tamagotchi)
        self.achievement = achievement
        self.description = description

class TamagotchiDeath(TamagotchiMessage):
    """Sent when tamagotchi dies"""
    
    def __init__(
        self,
        tamagotchi: 'BaseTamagotchi',
        cause: str,
        age: float
    ):
        super().__init__(tamagotchi)
        self.cause = cause
        self.age = age
```

## Customization Guide

### Creating Custom Personalities

```python
from tldw_chatbook.Widgets.Tamagotchi import Personality, register_personality

# Define a custom personality
vampire_personality = Personality(
    name='vampire',
    happiness_decay=-0.3,  # Doesn't need much social interaction
    hunger_increase=2.0,   # Gets hungry quickly (for blood!)
    energy_decay=0.1,      # Gains energy at night
    social_need=0.5,       # Prefers solitude
    resilience=1.5         # Hard to kill
)

# Register it
register_personality('vampire', vampire_personality)

# Use it
tamagotchi = Tamagotchi(
    name="Vlad",
    personality="vampire",
    sprite_theme="gothic"  # Custom sprite theme
)
```

### Custom Sprite Themes

```python
from tldw_chatbook.Widgets.Tamagotchi import SpriteManager

# Create a robot theme
robot_sprites = {
    'happy': ['[^.^]', '[*.*]', '[o.o]'],
    'sad': ['[T.T]', '[;.;]', '[x.x]'],
    'hungry': ['[?.?]', '[!.!]', '[@.@]'],
    'sleepy': ['[-.-]', '[z.z]', '[_.._]'],
    'sick': ['[%.%]', '[#.#]', '[&.&]'],
}

sprite_manager = SpriteManager(theme="custom")
for mood, sprites in robot_sprites.items():
    sprite_manager.register_sprite(mood, sprites)
```

### Advanced Storage Implementation

```python
class CloudStorage(StorageAdapter):
    """Example cloud storage implementation"""
    
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.session = httpx.Client()
    
    async def load(self, pet_id: str) -> Optional[Dict[str, Any]]:
        """Load from cloud"""
        response = await self.session.get(
            f"{self.endpoint}/pets/{pet_id}",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        if response.status_code == 200:
            return response.json()
        return None
    
    async def save(self, pet_id: str, state: Dict[str, Any]) -> bool:
        """Save to cloud"""
        response = await self.session.put(
            f"{self.endpoint}/pets/{pet_id}",
            json=state,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.status_code == 200
```

## Integration Examples

### 1. Status Bar Integration

```python
from textual.containers import Horizontal
from textual.widgets import Static
from tldw_chatbook.Widgets.Tamagotchi import CompactTamagotchi

class EnhancedFooter(Horizontal):
    """Footer with integrated tamagotchi"""
    
    DEFAULT_CSS = """
    EnhancedFooter {
        height: 1;
        dock: bottom;
        background: $surface;
        padding: 0 1;
    }
    
    EnhancedFooter Static {
        margin: 0 1;
    }
    
    EnhancedFooter CompactTamagotchi {
        margin: 0 2;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("Ready", id="status")
        yield Static("", id="spacer")  # Pushes pet to right
        yield CompactTamagotchi(
            name="Byte",
            personality="energetic",
            size="minimal"
        )
        yield Static("", id="metrics")
```

### 2. Sidebar Widget

```python
from textual.containers import VerticalScroll
from textual.widgets import Button
from tldw_chatbook.Widgets.Tamagotchi import Tamagotchi

class TamagotchiPanel(VerticalScroll):
    """Full tamagotchi panel for sidebar"""
    
    DEFAULT_CSS = """
    TamagotchiPanel {
        width: 30;
        dock: right;
        background: $panel;
        padding: 1;
    }
    
    TamagotchiPanel Button {
        width: 100%;
        margin: 1 0;
    }
    """
    
    def compose(self) -> ComposeResult:
        self.tamagotchi = Tamagotchi(
            name="Pixel",
            personality="needy",
            size="normal"
        )
        yield self.tamagotchi
        
        yield Button("Feed", id="feed-btn", variant="primary")
        yield Button("Play", id="play-btn", variant="success")
        yield Button("Sleep", id="sleep-btn", variant="warning")
        yield Button("Clean", id="clean-btn")
    
    @on(Button.Pressed, "#feed-btn")
    def feed_pet(self) -> None:
        self.tamagotchi.interact("feed")
    
    @on(Button.Pressed, "#play-btn")
    def play_with_pet(self) -> None:
        self.tamagotchi.interact("play")
    
    @on(Button.Pressed, "#sleep-btn")
    def put_to_sleep(self) -> None:
        self.tamagotchi.interact("sleep")
    
    @on(Button.Pressed, "#clean-btn")
    def clean_pet(self) -> None:
        self.tamagotchi.interact("clean")
```

### 3. Multi-Pet System

```python
from textual.containers import Grid
from tldw_chatbook.Widgets.Tamagotchi import Tamagotchi, SQLiteStorage

class PetCollection(Grid):
    """Manage multiple tamagotchis"""
    
    DEFAULT_CSS = """
    PetCollection {
        grid-size: 3 2;
        grid-gutter: 1;
        padding: 1;
    }
    
    PetCollection Tamagotchi {
        height: 5;
        border: round $primary;
    }
    """
    
    def compose(self) -> ComposeResult:
        storage = SQLiteStorage("~/.config/myapp/pets.db")
        
        pets = [
            ("Bit", "energetic"),
            ("Byte", "lazy"),
            ("Pixel", "balanced"),
            ("Vector", "needy"),
            ("Sprite", "balanced"),
            ("Raster", "energetic")
        ]
        
        for name, personality in pets:
            yield Tamagotchi(
                name=name,
                personality=personality,
                storage=storage,
                id=f"pet-{name.lower()}"
            )
    
    def on_tamagotchi_interaction(self, event: TamagotchiInteraction) -> None:
        """Handle interactions from any pet"""
        self.notify(f"{event.pet_name} was {event.action}!")
```

### 4. Notification Integration

```python
from textual.widgets import Static
from datetime import datetime

class TamagotchiWithNotifications(Tamagotchi):
    """Extended tamagotchi with notification support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notification_threshold = {
            'happiness': 20,
            'hunger': 80,
            'energy': 20,
            'health': 30
        }
    
    def watch_happiness(self, old_value: float, new_value: float) -> None:
        """Watch for happiness changes"""
        if new_value < self.notification_threshold['happiness']:
            self.app.notify(
                f"{self.name} is feeling sad! 😢",
                severity="warning"
            )
    
    def watch_hunger(self, old_value: float, new_value: float) -> None:
        """Watch for hunger changes"""
        if new_value > self.notification_threshold['hunger']:
            self.app.notify(
                f"{self.name} is very hungry! 🍽️",
                severity="warning"
            )
    
    def watch_energy(self, old_value: float, new_value: float) -> None:
        """Watch for energy changes"""
        if new_value < self.notification_threshold['energy']:
            self.app.notify(
                f"{self.name} needs sleep! 😴",
                severity="info"
            )
    
    def watch_health(self, old_value: float, new_value: float) -> None:
        """Watch for health changes"""
        if new_value < self.notification_threshold['health']:
            self.app.notify(
                f"{self.name} is sick! 🤒",
                severity="error"
            )
```

## Best Practices

### Performance Optimization

1. **Use Line API for Frequent Updates**
   ```python
   def render_line(self, y: int) -> Strip:
       """Efficient line-based rendering"""
       if y == 0:
           return Strip([Segment(self.sprite + " " + self.name)])
       elif y == 1:
           return Strip([Segment(f"H:{self.happiness} F:{self.hunger}")])
       return Strip()
   ```

2. **Batch State Updates**
   ```python
   def update_stats(self, **changes):
       """Update multiple stats at once"""
       with self.batch_update():
           for stat, value in changes.items():
               setattr(self, stat, value)
   ```

3. **Throttle Animations**
   ```python
   @throttle(0.1)  # Max 10 updates per second
   def animate_sprite(self):
       """Throttled animation updates"""
       self.sprite = next(self.animation_frames)
   ```

### State Management

1. **Use Reactive Properties Wisely**
   - Set `layout=False` for properties that don't affect size
   - Use `recompose=True` only when structure changes
   - Batch related updates

2. **Implement State Validation**
   ```python
   def validate_happiness(self, value: float) -> float:
       """Ensure happiness stays in valid range"""
       return max(0, min(100, value))
   ```

3. **Handle State Persistence Gracefully**
   ```python
   def _save_state(self) -> None:
       """Save with error handling"""
       try:
           state = {
               'happiness': self.happiness,
               'hunger': self.hunger,
               'energy': self.energy,
               'health': self.health,
               'age': self.age
           }
           self.storage.save(self.id, state)
       except Exception as e:
           self.log.error(f"Failed to save state: {e}")
   ```

### Testing Strategy

1. **Unit Tests**
   ```python
   def test_stat_decay():
       """Test stat decay over time"""
       engine = BehaviorEngine("balanced")
       changes = engine.calculate_decay(60)  # 1 minute
       assert changes['happiness'] == -0.5
       assert changes['hunger'] == 1.0
   ```

2. **Integration Tests**
   ```python
   async def test_tamagotchi_interaction():
       """Test full interaction cycle"""
       app = TamagotchiTestApp()
       async with app.run_test() as pilot:
           tamagotchi = app.query_one(Tamagotchi)
           initial_happiness = tamagotchi.happiness
           
           await pilot.click(tamagotchi)
           assert tamagotchi.happiness > initial_happiness
   ```

3. **Mock Timers for Testing**
   ```python
   class MockTimer:
       """Mock timer for predictable testing"""
       def __init__(self):
           self.callbacks = []
       
       def set_interval(self, interval, callback):
           self.callbacks.append((interval, callback))
           return self
       
       def trigger(self):
           for _, callback in self.callbacks:
               callback()
   ```

## Advanced Features

### Evolution System

```python
class EvolvingTamagotchi(BaseTamagotchi):
    """Tamagotchi that evolves through life stages"""
    
    stage = reactive("egg")
    evolution_points = reactive(0)
    
    EVOLUTION_STAGES = {
        'egg': {'next': 'baby', 'required_age': 0.5, 'required_points': 10},
        'baby': {'next': 'child', 'required_age': 2, 'required_points': 50},
        'child': {'next': 'teen', 'required_age': 5, 'required_points': 100},
        'teen': {'next': 'adult', 'required_age': 10, 'required_points': 200},
        'adult': {'next': None, 'required_age': None, 'required_points': None}
    }
    
    def check_evolution(self) -> None:
        """Check if ready to evolve"""
        current = self.EVOLUTION_STAGES[self.stage]
        if current['next'] is None:
            return
            
        if (self.age >= current['required_age'] and 
            self.evolution_points >= current['required_points']):
            self.evolve(current['next'])
    
    def evolve(self, new_stage: str) -> None:
        """Evolve to next stage"""
        old_stage = self.stage
        self.stage = new_stage
        
        # Update sprite theme for new stage
        self.sprite_manager.set_stage(new_stage)
        
        # Post evolution message
        self.post_message(
            TamagotchiEvolution(self, old_stage, new_stage)
        )
        
        # Play evolution animation
        self.animate(
            "opacity", 
            value=0.0, 
            duration=0.5,
            on_complete=lambda: self.animate("opacity", value=1.0, duration=0.5)
        )
```

### Achievement System

```python
class AchievementTracker:
    """Track and unlock achievements"""
    
    ACHIEVEMENTS = {
        'first_feed': {
            'name': 'First Meal',
            'description': 'Fed your pet for the first time',
            'condition': lambda stats: stats['total_feeds'] >= 1
        },
        'happy_pet': {
            'name': 'Joy Bringer',
            'description': 'Reached 100% happiness',
            'condition': lambda stats: stats['max_happiness'] >= 100
        },
        'survivor': {
            'name': 'Survivor',
            'description': 'Kept pet alive for 24 hours',
            'condition': lambda stats: stats['age'] >= 24
        },
        'caretaker': {
            'name': 'Dedicated Caretaker',
            'description': 'Interacted 100 times',
            'condition': lambda stats: stats['total_interactions'] >= 100
        }
    }
    
    def __init__(self):
        self.unlocked = set()
        self.stats = {
            'total_feeds': 0,
            'total_plays': 0,
            'total_interactions': 0,
            'max_happiness': 0,
            'age': 0
        }
    
    def check_achievements(self) -> List[str]:
        """Check for newly unlocked achievements"""
        newly_unlocked = []
        
        for achievement_id, achievement in self.ACHIEVEMENTS.items():
            if achievement_id not in self.unlocked:
                if achievement['condition'](self.stats):
                    self.unlocked.add(achievement_id)
                    newly_unlocked.append(achievement_id)
        
        return newly_unlocked
```

### Mini-Games

```python
class TamagotchiMiniGame(Static):
    """Base class for mini-games"""
    
    def __init__(self, tamagotchi: BaseTamagotchi):
        super().__init__()
        self.tamagotchi = tamagotchi
        self.score = 0
        self.playing = False
    
    def start_game(self) -> None:
        """Start the mini-game"""
        self.playing = True
        self.score = 0
        self.refresh()
    
    def end_game(self) -> None:
        """End the game and apply rewards"""
        self.playing = False
        
        # Reward based on score
        happiness_bonus = min(20, self.score * 2)
        self.tamagotchi.happiness += happiness_bonus
        
        self.post_message(
            GameCompleted(self, self.score, happiness_bonus)
        )

class CatchGame(TamagotchiMiniGame):
    """Simple catching mini-game"""
    
    def __init__(self, tamagotchi: BaseTamagotchi):
        super().__init__(tamagotchi)
        self.target_position = 5
        self.pet_position = 5
        
    def on_key(self, event: events.Key) -> None:
        """Handle arrow keys"""
        if not self.playing:
            return
            
        if event.key == "left":
            self.pet_position = max(0, self.pet_position - 1)
        elif event.key == "right":
            self.pet_position = min(10, self.pet_position + 1)
        elif event.key == "space":
            if self.pet_position == self.target_position:
                self.score += 1
                self.target_position = random.randint(0, 10)
        
        self.refresh()
    
    def render(self) -> str:
        """Render the game field"""
        if not self.playing:
            return f"Press ENTER to play! Score: {self.score}"
        
        field = ['.'] * 11
        field[self.target_position] = '🎯'
        field[self.pet_position] = self.tamagotchi.sprite
        
        return ''.join(field) + f"\nScore: {self.score}"
```

### Context Menu Actions

```python
from textual.widgets import Menu, MenuItem

class TamagotchiWithMenu(BaseTamagotchi):
    """Tamagotchi with right-click context menu"""
    
    def on_right_click(self, event: events.Click) -> None:
        """Show context menu on right-click"""
        menu_items = [
            MenuItem("Feed", action=lambda: self.interact("feed")),
            MenuItem("Play", action=lambda: self.interact("play")),
            MenuItem("Pet", action=lambda: self.interact("pet")),
            MenuItem("Clean", action=lambda: self.interact("clean")),
            MenuItem("-"),  # Separator
            MenuItem("Check Stats", action=self.show_stats),
            MenuItem("View Achievements", action=self.show_achievements),
            MenuItem("-"),
            MenuItem("Settings", action=self.show_settings)
        ]
        
        self.app.push_screen(
            ContextMenu(menu_items, position=(event.x, event.y))
        )
```

## Troubleshooting

### Common Issues

1. **Pet Not Saving State**
   - Check storage permissions
   - Verify storage path exists
   - Ensure proper shutdown handling

2. **Animations Not Working**
   - Verify timer is started
   - Check CSS animation support
   - Ensure widget is mounted

3. **High CPU Usage**
   - Increase update interval
   - Use Line API for rendering
   - Disable unnecessary animations

4. **Pet Dies Too Quickly**
   - Adjust personality settings
   - Increase initial stats
   - Reduce decay rates

### Debug Mode

```python
class DebugTamagotchi(BaseTamagotchi):
    """Tamagotchi with debug features"""
    
    def __init__(self, *args, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = debug
        
    def render(self) -> str:
        """Include debug info in render"""
        output = super().render()
        
        if self.debug:
            debug_info = (
                f"\nDEBUG: H:{self.happiness:.1f} "
                f"F:{self.hunger:.1f} E:{self.energy:.1f} "
                f"HP:{self.health:.1f} Age:{self.age:.2f}h"
            )
            output += debug_info
            
        return output
    
    def on_key(self, event: events.Key) -> None:
        """Debug keyboard shortcuts"""
        if not self.debug:
            return
            
        # Debug stat manipulation
        if event.key == "ctrl+h":
            self.happiness = 100
        elif event.key == "ctrl+f":
            self.hunger = 0
        elif event.key == "ctrl+e":
            self.energy = 100
        elif event.key == "ctrl+k":
            self.health = 0  # Kill pet
```

### Performance Profiling

```python
import time
from functools import wraps

def profile_method(func):
    """Decorator to profile method performance"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        duration = time.perf_counter() - start
        
        if duration > 0.1:  # Log slow operations
            self.log.warning(
                f"{func.__name__} took {duration:.3f}s"
            )
        
        return result
    return wrapper

class ProfiledTamagotchi(BaseTamagotchi):
    """Tamagotchi with performance profiling"""
    
    @profile_method
    def render(self) -> str:
        return super().render()
    
    @profile_method
    def _periodic_update(self) -> None:
        return super()._periodic_update()
```

## Module Structure Summary

The complete module structure provides:

1. **Core Components**: Base widget, sprites, behaviors, storage, messages
2. **Customization**: Personalities, themes, storage backends
3. **Integration**: Multiple integration patterns for different use cases
4. **Advanced Features**: Evolution, achievements, mini-games
5. **Developer Tools**: Debug mode, profiling, comprehensive testing

This modular architecture ensures the tamagotchi system can be:
- Easily integrated into any Textual application
- Customized for different themes and behaviors
- Extended with new features without modifying core code
- Tested thoroughly with mock components
- Optimized for performance in TUI environments

The system follows Textual best practices and provides a complete, production-ready virtual pet implementation.