"""
Tab Initializers - Modular handlers for tab initialization logic.

This package provides a clean architecture for handling tab-specific
initialization when tabs are shown or hidden in the application.
"""

from .base_initializer import BaseTabInitializer, TabInitializerRegistry
from .chat_tab_initializer import ChatTabInitializer
from .notes_tab_initializer import NotesTabInitializer
from .misc_tab_initializers import (
    CCPTabInitializer,
    MediaTabInitializer,
    SearchTabInitializer,
    IngestTabInitializer,
    ToolsSettingsTabInitializer,
    LLMTabInitializer,
    EvalsTabInitializer,
)

__all__ = [
    'BaseTabInitializer',
    'TabInitializerRegistry',
    'ChatTabInitializer',
    'NotesTabInitializer',
    'CCPTabInitializer',
    'MediaTabInitializer',
    'SearchTabInitializer',
    'IngestTabInitializer',
    'ToolsSettingsTabInitializer',
    'LLMTabInitializer',
    'EvalsTabInitializer',
]