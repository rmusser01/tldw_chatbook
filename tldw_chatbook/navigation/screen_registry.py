"""
Registry of all available screens in the application.
"""

from typing import Dict, Type, Optional
from textual.screen import Screen
from loguru import logger


class ScreenRegistry:
    """Central registry for all application screens."""
    
    def __init__(self):
        self._screens: Dict[str, Type[Screen]] = {}
        self._aliases: Dict[str, str] = {}
        self._load_screens()
    
    def _load_screens(self) -> None:
        """Load all screen classes."""
        # Import all screen classes
        from ..UI.Screens.chat_screen import ChatScreen
        from ..UI.Screens.media_ingest_screen import MediaIngestScreen
        from ..UI.Screens.coding_screen import CodingScreen
        from ..UI.Screens.conversation_screen import ConversationScreen
        from ..UI.Screens.media_screen import MediaScreen
        from ..UI.Screens.notes_screen import NotesScreen
        from ..UI.Screens.search_screen import SearchScreen
        from ..UI.Screens.evals_screen import EvalsScreen
        from ..UI.Screens.tools_settings_screen import ToolsSettingsScreen
        from ..UI.Screens.llm_screen import LLMScreen
        from ..UI.Screens.customize_screen import CustomizeScreen
        from ..UI.Screens.logs_screen import LogsScreen
        from ..UI.Screens.stats_screen import StatsScreen
        from ..UI.Screens.stts_screen import STTSScreen
        from ..UI.Screens.study_screen import StudyScreen
        from ..UI.Screens.chatbooks_screen import ChatbooksScreen
        from ..UI.Screens.subscription_screen import SubscriptionScreen
        
        # Register screens
        self._screens = {
            'chat': ChatScreen,
            'ingest': MediaIngestScreen,
            'coding': CodingScreen,
            'conversation': ConversationScreen,
            'media': MediaScreen,
            'notes': NotesScreen,
            'search': SearchScreen,
            'evals': EvalsScreen,
            'tools_settings': ToolsSettingsScreen,
            'llm': LLMScreen,
            'customize': CustomizeScreen,
            'logs': LogsScreen,
            'stats': StatsScreen,
            'stts': STTSScreen,
            'study': StudyScreen,
            'chatbooks': ChatbooksScreen,
            'subscription': SubscriptionScreen,
        }
        
        # Register aliases
        self._aliases = {
            'ccp': 'conversation',  # Conv/Char/Prompts alias
            'subscriptions': 'subscription',  # Plural alias
            'llm_management': 'llm',  # Legacy name
            'tools': 'tools_settings',  # Short name
            'settings': 'tools_settings',  # Alternative name
        }
        
        logger.info(f"Registered {len(self._screens)} screens with {len(self._aliases)} aliases")
    
    def get_screen_class(self, name: str) -> Optional[Type[Screen]]:
        """Get a screen class by name or alias."""
        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]
        
        return self._screens.get(name)
    
    def register_screen(self, name: str, screen_class: Type[Screen]) -> None:
        """Register a new screen."""
        self._screens[name] = screen_class
        logger.debug(f"Registered screen: {name} -> {screen_class.__name__}")
    
    def register_alias(self, alias: str, screen_name: str) -> None:
        """Register an alias for a screen."""
        if screen_name in self._screens:
            self._aliases[alias] = screen_name
            logger.debug(f"Registered alias: {alias} -> {screen_name}")
        else:
            logger.warning(f"Cannot register alias {alias}: screen {screen_name} not found")
    
    def list_screens(self) -> Dict[str, str]:
        """List all available screens."""
        return {
            name: cls.__name__ 
            for name, cls in self._screens.items()
        }
    
    def list_aliases(self) -> Dict[str, str]:
        """List all screen aliases."""
        return self._aliases.copy()
    
    def is_valid_screen(self, name: str) -> bool:
        """Check if a screen name or alias is valid."""
        return name in self._screens or name in self._aliases