"""
Textual Tamagotchi Module

A modular, customizable tamagotchi widget system for Textual applications.
Provides virtual pet functionality with customizable behaviors, sprites, and storage.
"""

from .base_tamagotchi import BaseTamagotchi, CompactTamagotchi, Tamagotchi
from .tamagotchi_behaviors import BehaviorEngine, Personality, PERSONALITIES, register_personality
from .tamagotchi_sprites import SpriteManager
from .tamagotchi_storage import StorageAdapter, JSONStorage, SQLiteStorage, MemoryStorage
from .tamagotchi_messages import (
    TamagotchiMessage,
    TamagotchiInteraction,
    TamagotchiStateChange,
    TamagotchiEvolution,
    TamagotchiAchievement,
    TamagotchiDeath
)
from .validators import (
    TamagotchiValidator,
    StateValidator,
    RateLimiter,
    ValidationError
)

__all__ = [
    # Core widget classes
    'BaseTamagotchi',
    'CompactTamagotchi',
    'Tamagotchi',
    
    # Behavior system
    'BehaviorEngine',
    'Personality',
    'PERSONALITIES',
    'register_personality',
    
    # Sprite system
    'SpriteManager',
    
    # Storage adapters
    'StorageAdapter',
    'JSONStorage',
    'SQLiteStorage',
    'MemoryStorage',
    
    # Messages
    'TamagotchiMessage',
    'TamagotchiInteraction',
    'TamagotchiStateChange',
    'TamagotchiEvolution',
    'TamagotchiAchievement',
    'TamagotchiDeath',
    
    # Validators
    'TamagotchiValidator',
    'StateValidator',
    'RateLimiter',
    'ValidationError',
]

__version__ = '1.0.0'