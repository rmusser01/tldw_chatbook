"""Application screens for screen-based navigation.

Screen classes are exported lazily so importing lightweight screen-adjacent
modules does not import every destination screen during app startup.
"""

from __future__ import annotations

from importlib import import_module


_SCREEN_EXPORTS = {
    "ChatScreen": ".chat_screen",
    "MediaIngestScreen": ".media_ingest_screen",
    "CodingScreen": ".coding_screen",
    "ConversationScreen": ".conversation_screen",
    "MediaScreen": ".media_screen",
    "NotesScreen": ".notes_screen",
    "SearchScreen": ".search_screen",
    "EvalsScreen": ".evals_screen",
    "ToolsSettingsScreen": ".tools_settings_screen",
    "LLMScreen": ".llm_screen",
    "CustomizeScreen": ".customize_screen",
    "LogsScreen": ".logs_screen",
    "StatsScreen": ".stats_screen",
}

__all__ = [
    'ChatScreen',
    'MediaIngestScreen',
    'CodingScreen',
    'ConversationScreen',
    'MediaScreen',
    'NotesScreen',
    'SearchScreen',
    'EvalsScreen',
    'ToolsSettingsScreen',
    'LLMScreen',
    'CustomizeScreen',
    'LogsScreen',
    'StatsScreen',
]


def __getattr__(name: str):
    if name not in _SCREEN_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_SCREEN_EXPORTS[name], __name__)
    screen_class = getattr(module, name)
    globals()[name] = screen_class
    return screen_class
