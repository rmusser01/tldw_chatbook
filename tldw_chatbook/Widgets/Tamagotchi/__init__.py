"""Compatibility-preserving lazy public exports for recovery isolation."""

from importlib import import_module

_EXPORTS = {
    "BaseTamagotchi": (".base_tamagotchi", "BaseTamagotchi"),
    "CompactTamagotchi": (".base_tamagotchi", "CompactTamagotchi"),
    "Tamagotchi": (".base_tamagotchi", "Tamagotchi"),
    "BehaviorEngine": (".tamagotchi_behaviors", "BehaviorEngine"),
    "Personality": (".tamagotchi_behaviors", "Personality"),
    "PERSONALITIES": (".tamagotchi_behaviors", "PERSONALITIES"),
    "register_personality": (".tamagotchi_behaviors", "register_personality"),
    "SpriteManager": (".tamagotchi_sprites", "SpriteManager"),
    "StorageAdapter": (".tamagotchi_storage", "StorageAdapter"),
    "JSONStorage": (".tamagotchi_storage", "JSONStorage"),
    "SQLiteStorage": (".tamagotchi_storage", "SQLiteStorage"),
    "MemoryStorage": (".tamagotchi_storage", "MemoryStorage"),
    "TamagotchiMessage": (".tamagotchi_messages", "TamagotchiMessage"),
    "TamagotchiInteraction": (".tamagotchi_messages", "TamagotchiInteraction"),
    "TamagotchiStateChange": (".tamagotchi_messages", "TamagotchiStateChange"),
    "TamagotchiEvolution": (".tamagotchi_messages", "TamagotchiEvolution"),
    "TamagotchiAchievement": (".tamagotchi_messages", "TamagotchiAchievement"),
    "TamagotchiDeath": (".tamagotchi_messages", "TamagotchiDeath"),
    "TamagotchiValidator": (".validators", "TamagotchiValidator"),
    "StateValidator": (".validators", "StateValidator"),
    "RateLimiter": (".validators", "RateLimiter"),
    "ValidationError": (".validators", "ValidationError"),
}

__all__ = [
    "BaseTamagotchi",
    "CompactTamagotchi",
    "Tamagotchi",
    "BehaviorEngine",
    "Personality",
    "PERSONALITIES",
    "register_personality",
    "SpriteManager",
    "StorageAdapter",
    "JSONStorage",
    "SQLiteStorage",
    "MemoryStorage",
    "TamagotchiMessage",
    "TamagotchiInteraction",
    "TamagotchiStateChange",
    "TamagotchiEvolution",
    "TamagotchiAchievement",
    "TamagotchiDeath",
    "TamagotchiValidator",
    "StateValidator",
    "RateLimiter",
    "ValidationError",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value


__version__ = "1.0.0"
