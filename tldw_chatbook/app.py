# tldw_cli - Textual CLI for LLMs
# Description: This file contains the main application logic for the tldw_cli, a Textual-based CLI for interacting with various LLM APIs.
#
# Disable progress bars early to prevent interference with TUI
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Disable Textual logging in production
# Set to a path to enable logging for debugging: os.environ['TEXTUAL_LOG'] = '/tmp/textual.log'
if 'TEXTUAL_LOG' not in os.environ:
    os.environ['TEXTUAL_LOG'] = ''  # Empty string disables logging

# Imports
import concurrent.futures
import functools
import inspect
import logging
import logging.handlers
import random
import subprocess
import sys
import threading
import time
import traceback
from typing import Union, Optional, Any, Dict, List, Callable
from textual.widget import Widget
#
# 3rd-Party Libraries
import asyncio
from PIL import Image
from loguru import logger as loguru_logger, logger
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import (
    Static, Button, Input, RichLog, TextArea, Select, ListView, Checkbox, Collapsible, ListItem, Label, Switch, Markdown
)
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from textual.worker import Worker
from textual.binding import Binding
from textual.message import Message
from textual.timer import Timer
from textual.css.query import QueryError
from textual.command import Hit, Hits, Provider
from functools import partial
from pathlib import Path

from tldw_chatbook.Utils.text import slugify
from tldw_chatbook.css.Themes.themes import ALL_THEMES
# from tldw_chatbook.css.css_loader import load_modular_css  # Removed - reverting to original CSS
from tldw_chatbook.Metrics.metrics import log_histogram, log_counter, log_resource_usage, init_metrics_server
from tldw_chatbook.Metrics.Otel_Metrics import init_metrics as init_otel_metrics
#
# --- Local API library Imports ---
from .Event_Handlers.LLM_Management_Events import (llm_management_events, llm_management_events_mlx_lm,
    llm_management_events_ollama, llm_management_events_onnx, llm_management_events_transformers,
                                                   llm_management_events_vllm)
from tldw_chatbook.Event_Handlers.Chat_Events.chat_streaming_events import handle_streaming_chunk, handle_stream_done
from tldw_chatbook.Event_Handlers.worker_events import StreamingChunk, StreamDone
from .Widgets.AppFooterStatus import AppFooterStatus
from .config import (
    get_media_db_path,
    get_prompts_db_path,
)
from .Logging_Config import configure_application_logging
from tldw_chatbook.Constants import ALL_TABS, TAB_CCP, TAB_CHAT, TAB_LOGS, TAB_NOTES, TAB_STATS, TAB_TOOLS_SETTINGS, TAB_CUSTOMIZE, \
    TAB_INGEST, TAB_LLM, TAB_MEDIA, TAB_SEARCH, TAB_EVALS, LLAMA_CPP_SERVER_ARGS_HELP_TEXT, \
    LLAMAFILE_SERVER_ARGS_HELP_TEXT, TAB_CODING, TAB_STTS, TAB_STUDY, TAB_SUBSCRIPTIONS, TAB_CHATBOOKS
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.config import CLI_APP_CLIENT_ID
from tldw_chatbook.Logging_Config import RichLogHandler
from tldw_chatbook.Prompt_Management import Prompts_Interop as prompts_interop
from tldw_chatbook.Utils.Emoji_Handling import get_char, EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN, supports_emoji
from tldw_chatbook.Utils.log_widget_manager import LogWidgetManager
from tldw_chatbook.Utils.ui_helpers import UIHelpers
from tldw_chatbook.Utils.db_status_manager import DBStatusManager
from tldw_chatbook.Event_Handlers.worker_handlers import (
    WorkerHandlerRegistry, ChatWorkerHandler, ServerWorkerHandler,
    AIGenerationHandler, MiscWorkerHandler
)
from .config import (
    CONFIG_TOML_CONTENT,
    DEFAULT_CONFIG_PATH,
    load_settings,
    get_cli_setting,
    get_cli_providers_and_models,
    API_MODELS_BY_PROVIDER,
    LOCAL_PROVIDERS,
    load_cli_config_and_ensure_existence,
    set_encryption_password, )
from .Event_Handlers import (
    conv_char_events as ccp_handlers,
    notes_events as notes_handlers,
    worker_events, ingest_events,
    llm_nav_events, media_events, notes_events, app_lifecycle, tab_events,
    search_events, subscription_events,
)
from .Event_Handlers.Chat_Events import chat_events as chat_handlers, chat_events_sidebar, chat_events_worldbooks, \
    chat_events_dictionaries
from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSRequestEvent, TTSCompleteEvent, TTSPlaybackEvent, TTSProgressEvent, TTSEventHandler
)
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler, STTSPlaygroundGenerateEvent, STTSSettingsSaveEvent, STTSAudioBookGenerateEvent
)
from .Notes.Notes_Library import NotesInteropService
from .DB.ChaChaNotes_DB import CharactersRAGDBError, ConflictError
from tldw_chatbook.Widgets.Chat_Widgets.chat_message import ChatMessage
from tldw_chatbook.Widgets.Chat_Widgets.chat_message_enhanced import ChatMessageEnhanced
from tldw_chatbook.Widgets.Note_Widgets.notes_sidebar_left import NotesSidebarLeft
from tldw_chatbook.Widgets.Note_Widgets.notes_sidebar_right import NotesSidebarRight
from .Widgets.titlebar import TitleBar
from .Widgets.splash_screen import SplashScreen
from .LLM_Calls.LLM_API_Calls import (
        chat_with_openai, chat_with_anthropic, chat_with_cohere,
        chat_with_groq, chat_with_openrouter, chat_with_huggingface,
        chat_with_deepseek, chat_with_mistral, chat_with_google,
)
from .LLM_Calls.LLM_API_Calls_Local import (
    chat_with_llama, chat_with_kobold, chat_with_oobabooga,
    chat_with_vllm, chat_with_tabbyapi, chat_with_aphrodite,
    chat_with_ollama, chat_with_custom_openai, chat_with_custom_openai_2, chat_with_local_llm
)
from tldw_chatbook.config import get_chachanotes_db_path, settings, get_chachanotes_db_lazy
from .UI.Chat_Window import ChatWindow
from .UI.Chat_Window_Enhanced import ChatWindowEnhanced
from .UI.Conv_Char_Window import CCPWindow
from .UI.Notes_Window import NotesWindow
from .UI.Logs_Window import LogsWindow
from .UI.Stats_Window import StatsWindow
from .UI.NewIngestWindow import NewIngestWindow
from .UI.Ingest_Window import INGEST_NAV_BUTTON_IDS, INGEST_VIEW_IDS, MEDIA_TYPES
from .UI.Tools_Settings_Window import ToolsSettingsWindow
from .UI.LLM_Management_Window import LLMManagementWindow
from .UI.Customize_Window import CustomizeWindow
# Using unified Evals dashboard
from .UI.Evals_Window_v3_unified import EvalsWindow
from .UI.Coding_Window import CodingWindow
from .UI.STTS_Window import STTSWindow
from .UI.Study_Window import StudyWindow
from .UI.Chatbooks_Window import ChatbooksWindow
from .UI.Tab_Bar import TabBar
from .UI.Tab_Links import TabLinks
from .UI.Tab_Dropdown import TabDropdown
from .UI.MediaWindow_v2 import MediaWindow
from .UI.SearchWindow import SearchWindow
from .UI.SearchWindow import ( # Import new constants from SearchWindow.py
    SEARCH_VIEW_RAG_QA,
    SEARCH_NAV_RAG_QA,
    SEARCH_NAV_RAG_CHAT,
    SEARCH_NAV_RAG_MANAGEMENT,
    SEARCH_NAV_WEB_SEARCH,
    SEARCH_NAV_EMBEDDINGS_CREATE,
    SEARCH_NAV_EMBEDDINGS_MANAGE
)
API_IMPORTS_SUCCESSFUL = True

# Try to import SubscriptionWindow if dependencies are available
SubscriptionWindow = None
try:
    from .UI.SubscriptionWindow import SubscriptionWindow
    SUBSCRIPTIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Subscriptions feature unavailable: {e}")
    SUBSCRIPTIONS_AVAILABLE = False
#
#######################################################################################################################
#
# Statics

if API_IMPORTS_SUCCESSFUL:
    API_FUNCTION_MAP = {
        "OpenAI": chat_with_openai,
        "Anthropic": chat_with_anthropic,
        "Cohere": chat_with_cohere,
        "HuggingFace": chat_with_huggingface,
        "DeepSeek": chat_with_deepseek,
        "Google": chat_with_google, # Key from config
        "Groq": chat_with_groq,
        "koboldcpp": chat_with_kobold,  # Key from config
        "llama_cpp": chat_with_llama,  # Key from config
        "MistralAI": chat_with_mistral,  # Key from config
        "Oobabooga": chat_with_oobabooga,  # Key from config
        "OpenRouter": chat_with_openrouter,
        "vllm": chat_with_vllm,  # Key from config
        "TabbyAPI": chat_with_tabbyapi,  # Key from config
        "Aphrodite": chat_with_aphrodite,  # Key from config
        "Ollama": chat_with_ollama,  # Key from config
        "Custom": chat_with_custom_openai,  # Key from config
        "Custom_2": chat_with_custom_openai_2,  # Key from config
        "local-llm": chat_with_local_llm
    }
    logging.info(f"API_FUNCTION_MAP populated with {len(API_FUNCTION_MAP)} entries.")
else:
    API_FUNCTION_MAP = {}
    logging.error("API_FUNCTION_MAP is empty due to import failures.")

ALL_API_MODELS = {**API_MODELS_BY_PROVIDER, **LOCAL_PROVIDERS} # If needed for sidebar defaults
AVAILABLE_PROVIDERS = list(ALL_API_MODELS.keys()) # If needed
#
#
#####################################################################################################################
#
# Functions:

# --- Global variable for config ---
APP_CONFIG = load_settings()

# Early logging configuration removed - handled by configure_application_logging() during app initialization


class ThemeProvider(Provider):
    """A command provider for theme switching."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the ThemeProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        """Search for theme commands."""
        matcher = self.matcher(query)
        
        # Always show the main "Change Theme" command
        main_command_score = matcher.match("Theme: Change Theme")
        if main_command_score > 0:
            yield Hit(
                main_command_score,
                matcher.highlight("Theme: Change Theme"),
                partial(self.show_theme_submenu),
                help="Open theme selection menu"
            )
        
        # Only show individual themes if user is specifically searching for theme-related terms
        if any(term in query.lower() for term in ["switch", "theme", "dark", "light", "color", "solarized", "gruvbox", "dracula"]):
            # Get available theme names from registered themes
            available_themes = ["textual-dark", "textual-light"]  # Built-in themes
            # Add custom themes from ALL_THEMES
            for theme in ALL_THEMES:
                theme_name = theme.name if hasattr(theme, 'name') else str(theme)
                available_themes.append(theme_name)
            
            for theme_name in available_themes:
                command_text = f"Theme: Switch to {theme_name.replace('_', ' ').replace('-', ' ').title()}"
                score = matcher.match(command_text)
                if score > 0:
                    yield Hit(
                        score * 0.9,  # Slightly lower priority than main command
                        matcher.highlight(command_text),
                        partial(self.switch_theme, theme_name),
                        help=f"Change theme to {theme_name}"
                    )
    
    async def discover(self) -> Hits:
        """Show only the main theme command when palette is first opened."""
        yield Hit(
            1.0,
            "Theme: Change Theme",
            partial(self.show_theme_submenu),
            help="Open theme selection menu"
        )
    
    def show_theme_submenu(self) -> None:
        """Show a notification with instruction to search for themes."""
        self.app.notify("Type 'theme' in the command palette to see all available themes", severity="information")
    
    def switch_theme(self, theme_name: str) -> None:
        """Switch to the specified theme and save to config."""
        try:
            self.app.theme = theme_name
            self.app.notify(f"Theme changed to {theme_name}", severity="information")
            
            # Save the theme preference to config
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("general", "default_theme", theme_name)
            
        except Exception as e:
            self.app.notify(f"Failed to apply theme: {e}", severity="error")


class TabNavigationProvider(Provider):
    """Provider for tab navigation commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the TabNavigationProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        tab_commands = [
            ("Tab Navigation: Switch to Chat", TAB_CHAT, "Switch to the main chat interface"),
            ("Tab Navigation: Switch to Character Chat", TAB_CCP, "Switch to character and conversation management"),
            ("Tab Navigation: Switch to Notes", TAB_NOTES, "Switch to notes management"),
            ("Tab Navigation: Switch to Media", TAB_MEDIA, "Switch to media library"),
            ("Tab Navigation: Switch to Search", TAB_SEARCH, "Switch to search interface"),
            ("Tab Navigation: Switch to Ingest", TAB_INGEST, "Switch to content ingestion"),
            ("Tab Navigation: Switch to Tools & Settings", TAB_TOOLS_SETTINGS, "Switch to settings and configuration"),
            ("Tab Navigation: Switch to Customize", TAB_CUSTOMIZE, "Switch to appearance customization"),
            ("Tab Navigation: Switch to LLM Management", TAB_LLM, "Switch to LLM provider management"),
            ("Tab Navigation: Switch to Logs", TAB_LOGS, "Switch to application logs"),
            ("Tab Navigation: Switch to Stats", TAB_STATS, "Switch to statistics view"),
            ("Tab Navigation: Switch to Evaluations", TAB_EVALS, "Switch to evaluation tools"),
            ("Tab Navigation: Switch to Coding", TAB_CODING, "Switch to coding assistant"),
        ]
        
        for command_text, tab_id, help_text in tab_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.switch_tab, tab_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_tabs = [
            ("Tab Navigation: Switch to Chat", TAB_CHAT, "Switch to the main chat interface"),
            ("Tab Navigation: Switch to Character Chat", TAB_CCP, "Switch to character and conversation management"),
            ("Tab Navigation: Switch to Notes", TAB_NOTES, "Switch to notes management"),
            ("Tab Navigation: Switch to Search", TAB_SEARCH, "Switch to search interface"),
            ("Tab Navigation: Switch to Tools & Settings", TAB_TOOLS_SETTINGS, "Switch to settings and configuration"),
            ("Tab Navigation: Switch to Customize", TAB_CUSTOMIZE, "Switch to appearance customization"),
        ]
        
        for command_text, tab_id, help_text in popular_tabs:
            yield Hit(
                1.0,
                command_text,
                partial(self.switch_tab, tab_id),
                help=help_text
            )
    
    def switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab."""
        try:
            self.app.current_tab = tab_id
            self.app.notify(f"Switched to {tab_id.replace('_', ' ').title()} tab", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to switch tab: {e}", severity="error")


class LLMProviderProvider(Provider):
    """Provider for LLM provider management commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the LLMProviderProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        # Get available providers from the app
        available_providers = AVAILABLE_PROVIDERS if 'AVAILABLE_PROVIDERS' in globals() else []
        
        provider_commands = [
            ("LLM Provider Management: Show Current Provider", None, "Display currently selected LLM provider"),
            ("LLM Provider Management: Test API Connection", None, "Test connection to current LLM provider"),
        ]
        
        # Add provider switching commands
        for provider in available_providers:
            provider_name = provider.replace('_', ' ').title()
            command_text = f"LLM Provider Management: Switch to {provider_name}"
            provider_commands.append((command_text, provider, f"Switch to {provider_name} provider"))
        
        for command_text, provider_id, help_text in provider_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_llm_command, provider_id, command_text),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_providers = ["OpenAI", "Anthropic", "Cohere", "Groq", "Ollama"]
        
        yield Hit(
            1.0,
            "LLM Provider Management: Show Current Provider",
            partial(self.handle_llm_command, None, "show_current"),
            help="Display currently selected LLM provider"
        )
        
        for provider in popular_providers:
            yield Hit(
                0.9,
                f"LLM Provider Management: Switch to {provider}",
                partial(self.handle_llm_command, provider, f"switch_{provider}"),
                help=f"Switch to {provider} provider"
            )
    
    def handle_llm_command(self, provider_id: str, command: str) -> None:
        """Handle LLM provider commands."""
        try:
            if provider_id is None or "show_current" in command:
                # Show current provider
                current = getattr(self.app, 'current_provider', 'Unknown')
                self.app.notify(f"Current LLM provider: {current}", severity="information")
            elif "test" in command.lower():
                # Test API connection (placeholder)
                self.app.notify("API connection test initiated", severity="information")
            else:
                # Switch provider (placeholder - would need to integrate with actual provider switching logic)
                self.app.notify(f"Provider switch to {provider_id} requested", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute LLM command: {e}", severity="error")


class QuickActionsProvider(Provider):
    """Provider for quick action commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the QuickActionsProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        quick_actions = [
            ("Quick Actions: New Chat Conversation", "new_chat", "Start a new chat conversation"),
            ("Quick Actions: New Character Chat", "new_character", "Start a new character-based conversation"),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            ("Quick Actions: Clear Current Chat", "clear_chat", "Clear the current chat conversation"),
            ("Quick Actions: Export Chat as Markdown", "export_chat", "Export current chat to markdown file"),
            ("Quick Actions: Import Media File", "import_media", "Import a new media file for processing"),
            ("Quick Actions: Search All Content", "search_all", "Search across all content"),
            ("Quick Actions: Refresh Database", "refresh_db", "Refresh database connections"),
        ]
        
        for command_text, action_id, help_text in quick_actions:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.execute_quick_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_actions = [
            ("Quick Actions: New Chat Conversation", "new_chat", "Start a new chat conversation"),
            ("Quick Actions: New Note", "new_note", "Create a new note"),
            ("Quick Actions: Search All Content", "search_all", "Search across all content"),
            ("Quick Actions: Import Media File", "import_media", "Import a new media file for processing"),
        ]
        
        for command_text, action_id, help_text in popular_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.execute_quick_action, action_id),
                help=help_text
            )
    
    def execute_quick_action(self, action_id: str) -> None:
        """Execute the specified quick action."""
        try:
            if action_id == "new_chat":
                self.app.current_tab = TAB_CHAT
                self.app.notify("Switched to Chat tab for new conversation", severity="information")
            elif action_id == "new_character":
                self.app.current_tab = TAB_CCP
                self.app.notify("Switched to Character Chat tab", severity="information")
            elif action_id == "new_note":
                self.app.current_tab = TAB_NOTES
                self.app.notify("Switched to Notes tab for new note", severity="information")
            elif action_id == "search_all":
                self.app.current_tab = TAB_SEARCH
                self.app.notify("Switched to Search tab", severity="information")
            elif action_id == "import_media":
                self.app.current_tab = TAB_INGEST
                self.app.notify("Switched to Ingest tab for media import", severity="information")
            else:
                self.app.notify(f"Quick action '{action_id}' initiated", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute quick action: {e}", severity="error")


class SettingsProvider(Provider):
    """Provider for settings and preferences commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the SettingsProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        settings_commands = [
            ("Settings & Preferences: Open Config File", "open_config", "Open the configuration file for editing"),
            ("Settings & Preferences: Reload Configuration", "reload_config", "Reload configuration from file"),
            ("Settings & Preferences: Toggle Streaming Mode", "toggle_streaming", "Toggle LLM streaming mode on/off"),
            ("Settings & Preferences: Set Temperature to Low (0.1)", "temp_low", "Set LLM temperature to 0.1 for focused responses"),
            ("Settings & Preferences: Set Temperature to Medium (0.7)", "temp_med", "Set LLM temperature to 0.7 for balanced responses"),
            ("Settings & Preferences: Set Temperature to High (1.0)", "temp_high", "Set LLM temperature to 1.0 for creative responses"),
            ("Settings & Preferences: Reset to Default Settings", "reset_defaults", "Reset all settings to default values"),
            ("Settings & Preferences: Show Database Stats", "db_stats", "Show database size and statistics"),
            ("Settings & Preferences: Open Settings Tab", "open_settings", "Navigate to Tools & Settings tab"),
        ]
        
        for command_text, setting_id, help_text in settings_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_setting, setting_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_settings = [
            ("Settings & Preferences: Open Settings Tab", "open_settings", "Navigate to Tools & Settings tab"),
            ("Settings & Preferences: Open Config File", "open_config", "Open the configuration file for editing"),
            ("Settings & Preferences: Show Database Stats", "db_stats", "Show database size and statistics"),
            ("Settings & Preferences: Toggle Streaming Mode", "toggle_streaming", "Toggle LLM streaming mode on/off"),
        ]
        
        for command_text, setting_id, help_text in popular_settings:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_setting, setting_id),
                help=help_text
            )
    
    def handle_setting(self, setting_id: str) -> None:
        """Handle settings commands."""
        try:
            if setting_id == "open_settings":
                self.app.current_tab = TAB_TOOLS_SETTINGS
                self.app.notify("Opened Tools & Settings tab", severity="information")
            elif setting_id == "open_config":
                from .config import DEFAULT_CONFIG_PATH
                self.app.notify(f"Config file location: {DEFAULT_CONFIG_PATH}", severity="information")
            elif setting_id == "reload_config":
                self.app.notify("Configuration reload requested", severity="information")
            elif setting_id == "db_stats":
                self.app.notify("Database statistics display requested", severity="information")
            elif setting_id.startswith("temp_"):
                temp_map = {"temp_low": "0.1", "temp_med": "0.7", "temp_high": "1.0"}
                temp_value = temp_map.get(setting_id, "0.7")
                self.app.notify(f"Temperature set to {temp_value}", severity="information")
            else:
                self.app.notify(f"Settings action '{setting_id}' initiated", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute settings command: {e}", severity="error")


class CharacterProvider(Provider):
    """Provider for character and persona management commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the CharacterProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        character_commands = [
            ("Character/Persona Management: Create New Character", "new_character", "Create a new character or persona"),
            ("Character/Persona Management: Show All Characters", "list_characters", "Display all available characters"),
            ("Character/Persona Management: Switch Character", "switch_character", "Switch to a different character"),
            ("Character/Persona Management: Edit Current Character", "edit_character", "Edit the current character settings"),
            ("Character/Persona Management: Delete Character", "delete_character", "Delete a character (with confirmation)"),
            ("Character/Persona Management: Import Character", "import_character", "Import character from file"),
            ("Character/Persona Management: Export Character", "export_character", "Export character to file"),
            ("Character/Persona Management: Open Character Tab", "open_character_tab", "Navigate to Character Chat tab"),
        ]
        
        for command_text, action_id, help_text in character_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_character_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_character_actions = [
            ("Character/Persona Management: Open Character Tab", "open_character_tab", "Navigate to Character Chat tab"),
            ("Character/Persona Management: Create New Character", "new_character", "Create a new character or persona"),
            ("Character/Persona Management: Show All Characters", "list_characters", "Display all available characters"),
            ("Character/Persona Management: Switch Character", "switch_character", "Switch to a different character"),
        ]
        
        for command_text, action_id, help_text in popular_character_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_character_action, action_id),
                help=help_text
            )
    
    def handle_character_action(self, action_id: str) -> None:
        """Handle character management actions."""
        try:
            if action_id == "open_character_tab":
                self.app.current_tab = TAB_CCP
                self.app.notify("Opened Character Chat tab", severity="information")
            elif action_id == "new_character":
                self.app.current_tab = TAB_CCP
                self.app.notify("Navigate to Character Chat to create new character", severity="information")
            elif action_id == "list_characters":
                self.app.current_tab = TAB_CCP
                self.app.notify("Showing all characters in Character Chat tab", severity="information")
            else:
                self.app.notify(f"Character action '{action_id}' requested", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute character action: {e}", severity="error")


class MediaProvider(Provider):
    """Provider for media and content management commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the MediaProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        media_commands = [
            ("Media & Content: Open Media Library", "open_media", "Navigate to media library"),
            ("Media & Content: Recent Media Files", "recent_media", "Show recently added media files"),
            ("Media & Content: Search Transcripts", "search_transcripts", "Search through media transcripts"),
            ("Media & Content: Show Ingested Content", "show_ingested", "Display all ingested content"),
            ("Media & Content: Import New Media", "import_new", "Import new media file"),
            ("Media & Content: Open Media Database", "open_db", "View media database contents"),
            ("Media & Content: Refresh Media Library", "refresh_media", "Refresh media library"),
            ("Media & Content: Export Media List", "export_list", "Export media list to file"),
        ]
        
        for command_text, action_id, help_text in media_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_media_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_media_actions = [
            ("Media & Content: Open Media Library", "open_media", "Navigate to media library"),
            ("Media & Content: Import New Media", "import_new", "Import new media file"),
            ("Media & Content: Search Transcripts", "search_transcripts", "Search through media transcripts"),
            ("Media & Content: Recent Media Files", "recent_media", "Show recently added media files"),
        ]
        
        for command_text, action_id, help_text in popular_media_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_media_action, action_id),
                help=help_text
            )
    
    def handle_media_action(self, action_id: str) -> None:
        """Handle media management actions."""
        try:
            if action_id == "open_media":
                self.app.current_tab = TAB_MEDIA
                self.app.notify("Opened Media Library tab", severity="information")
            elif action_id == "import_new":
                self.app.current_tab = TAB_INGEST
                self.app.notify("Opened Ingest tab for media import", severity="information")
            elif action_id == "search_transcripts":
                self.app.current_tab = TAB_SEARCH
                self.app.notify("Opened Search tab for transcript search", severity="information")
            else:
                self.app.notify(f"Media action '{action_id}' requested", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute media action: {e}", severity="error")


class DeveloperProvider(Provider):
    """Provider for developer and debug commands."""
    
    def __init__(self, screen, *args, **kwargs):
        """Initialize the DeveloperProvider with required screen parameter."""
        super().__init__(screen, *args, **kwargs)
    
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        
        dev_commands = [
            ("Developer/Debug Commands: Show App Info", "app_info", "Display application version and build info"),
            ("Developer/Debug Commands: Open Log File", "open_logs", "Navigate to application logs"),
            ("Developer/Debug Commands: Clear Cache", "clear_cache", "Clear application cache"),
            ("Developer/Debug Commands: Show Keybindings", "show_keys", "Display all keyboard shortcuts"),
            ("Developer/Debug Commands: Debug Mode Toggle", "toggle_debug", "Toggle debug mode on/off"),
            ("Developer/Debug Commands: Memory Usage", "memory_usage", "Show current memory usage"),
            ("Developer/Debug Commands: Database Integrity Check", "db_check", "Check database integrity"),
            ("Developer/Debug Commands: Export Debug Info", "export_debug", "Export debug information to file"),
        ]
        
        for command_text, action_id, help_text in dev_commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    partial(self.handle_dev_action, action_id),
                    help=help_text
                )
    
    async def discover(self) -> Hits:
        popular_dev_actions = [
            ("Developer/Debug Commands: Open Log File", "open_logs", "Navigate to application logs"),
            ("Developer/Debug Commands: Show App Info", "app_info", "Display application version and build info"),
            ("Developer/Debug Commands: Show Keybindings", "show_keys", "Display all keyboard shortcuts"),
        ]
        
        for command_text, action_id, help_text in popular_dev_actions:
            yield Hit(
                1.0,
                command_text,
                partial(self.handle_dev_action, action_id),
                help=help_text
            )
    
    def handle_dev_action(self, action_id: str) -> None:
        """Handle developer/debug actions."""
        try:
            if action_id == "open_logs":
                self.app.current_tab = TAB_LOGS
                self.app.notify("Opened Logs tab", severity="information")
            elif action_id == "app_info":
                self.app.notify("tldw_chatbook - TUI for LLM interactions", severity="information")
            elif action_id == "show_keys":
                self.app.notify("Keybindings: Ctrl+Q (quit), Ctrl+P (palette)", severity="information")
            elif action_id == "clear_cache":
                self.app.notify("Cache clear requested", severity="information")
            else:
                self.app.notify(f"Developer action '{action_id}' initiated", severity="information")
        except Exception as e:
            self.app.notify(f"Failed to execute developer action: {e}", severity="error")


# --- Placeholder Window for Lazy Loading ---
class PlaceholderWindow(Container):
    """A lightweight placeholder that defers actual window creation until needed."""
    
    def __init__(self, app_instance: 'TldwCli', window_class: type, window_id: str, classes: str = "") -> None:
        """Initialize placeholder with window creation parameters."""
        super().__init__(id=window_id, classes=f"placeholder-window {classes}")
        self.app_instance = app_instance
        self.window_class = window_class
        self.window_id = window_id
        self.actual_classes = classes
        self._actual_window = None
        self._initialized = False
        # Log placeholder creation
        logger.debug(f"PlaceholderWindow created for {window_id} (class: {window_class.__name__})")
    
    def initialize(self) -> None:
        """Create and mount the actual window widget."""
        if self._initialized:
            return
            
        logger.info(f"Initializing actual window for {self.window_id}")
        start_time = time.perf_counter()
        
        try:
            # Remove the loading placeholder first
            for child in list(self.children):
                child.remove()
            
            # Create the actual window
            # EvalsLab is a Container that doesn't take app instance as first argument
            if self.window_class.__name__ == 'EvalsLab':
                self._actual_window = self.window_class(id=self.window_id, classes=self.actual_classes)
            else:
                self._actual_window = self.window_class(self.app_instance, id=self.window_id, classes=self.actual_classes)
            
            # Clear placeholder styling and mount actual window
            self.remove_class("placeholder-window")
            # Set proper layout for the container AND make it visible
            self.styles.layout = "vertical"
            self.styles.height = "100%"
            self.styles.width = "100%"
            self.styles.display = "block"  # CRITICAL: Reset display to block after removing placeholder class
            
            # Make sure the actual window fills the container
            self._actual_window.styles.height = "100%"
            self._actual_window.styles.width = "100%"
            
            self.mount(self._actual_window)
            self._initialized = True
            
            # Populate widgets for specific windows after initialization
            self._populate_window_widgets()
            
            # Log timing
            duration = time.perf_counter() - start_time
            log_histogram("lazy_window_initialization_seconds", duration,
                         labels={"window": self.window_id.replace("-window", "")},
                         documentation="Time to initialize lazy-loaded window")
            logger.info(f"Window {self.window_id} initialized in {duration:.3f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize window {self.window_id}: {str(e)}", exc_info=True)
            # Clear any existing children before showing error
            for child in list(self.children):
                child.remove()
            self.mount(Static(f"Error loading {self.window_id}: {str(e)}", classes="error"))
    
    def _populate_window_widgets(self) -> None:
        """Populate widgets for specific windows after they're initialized."""
        # Don't populate widgets here - let the watch_current_tab handle it
        # This prevents timing issues where widgets aren't ready yet
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if the actual window has been initialized."""
        return self._initialized
    
    def compose(self) -> ComposeResult:
        """Show a loading message until initialized."""
        if not self._initialized:
            yield Static("Loading...", classes="loading-placeholder")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Proxy button presses to the actual window if initialized."""
        if self._initialized and self._actual_window:
            if hasattr(self._actual_window, 'on_button_pressed'):
                result = self._actual_window.on_button_pressed(event)
                if hasattr(result, '__await__'):
                    await result


# --- Main App ---
class TldwCli(App[None]):  # Specify return type for run() if needed, None is common
    """A Textual app for interacting with LLMs."""
    #TITLE = "🧠📝🔍  tldw CLI"
    TITLE = "tldw chatbook"
    # CSS file path
    CSS_PATH = str(Path(__file__).parent / "css/tldw_cli_modular.tcss")
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit App", show=True),
        Binding("ctrl+p", "command_palette", "Palette Menu", show=True)
    ]
    COMMANDS = App.COMMANDS | {
        ThemeProvider,
        TabNavigationProvider,
        LLMProviderProvider,
        QuickActionsProvider,
        SettingsProvider,
        CharacterProvider,
        MediaProvider,
        DeveloperProvider
    }

    ALL_INGEST_VIEW_IDS = INGEST_VIEW_IDS
    ALL_MAIN_WINDOW_IDS = [ # Assuming these are your main content window IDs
        "chat-window", "conversations_characters_prompts-window", "notes-window",
        "ingest-window", "tools_settings-window", "llm_management-window",
        "media-window", "search-window", "logs-window", "stats-window", "evals-window", 
        "coding-window", "stts-window", "study-window", "chatbooks-window", "customize-window"
    ]

    # Define reactive at class level with a placeholder default and type hint
    current_tab: reactive[str] = reactive("")
    ccp_active_view: reactive[str] = reactive("conversation_details_view")
    
    # Splash screen state
    splash_screen_active: reactive[bool] = reactive(False)
    _splash_screen_widget: Optional[SplashScreen] = None

    # Add state to hold the currently streaming AI message widget
    # Use a lock to prevent race conditions when modifying shared state
    _chat_state_lock = threading.Lock()
    current_ai_message_widget: Optional[Union[ChatMessage, ChatMessageEnhanced]] = None
    current_chat_worker: Optional[Worker] = None
    current_chat_is_streaming: bool = False

    # --- REACTIVES FOR PROVIDER SELECTS ---
    # Initialize with a dummy value or fetch default from config here
    # Ensure the initial value matches what's set in compose/settings_sidebar
    # Fetching default provider from config:
    _default_chat_provider = APP_CONFIG.get("chat_defaults", {}).get("provider", "OpenAI")
    _default_ccp_provider = APP_CONFIG.get("character_defaults", {}).get("provider", "Anthropic") # Changed from character_defaults

    chat_api_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)
    # Renamed character_api_provider_value to ccp_api_provider_value for clarity with TAB_CCP
    ccp_api_provider_value: reactive[Optional[str]] = reactive(_default_ccp_provider)
    # RAG expansion provider reactive
    rag_expansion_provider_value: reactive[Optional[str]] = reactive(_default_chat_provider)

    # --- Reactives for CCP Character EDITOR (Center Pane) ---
    current_editing_character_id: reactive[Optional[str]] = reactive(None)
    current_editing_character_data: reactive[Optional[Dict[str, Any]]] = reactive(None)

    # DB Size checker - now using AppFooterStatus
    _db_size_status_widget: Optional[AppFooterStatus] = None
    # DB size update timer moved to DBStatusManager
    _token_count_update_timer: Optional[Timer] = None

    # Reactives for sidebar
    chat_sidebar_collapsed: reactive[bool] = reactive(False)
    chat_right_sidebar_collapsed: reactive[bool] = reactive(False)  # For character sidebar
    # Load saved width from config, default to 25% if not set
    _saved_width = settings.get("chat_defaults", {}).get("right_sidebar_width", 25)
    chat_right_sidebar_width: reactive[int] = reactive(_saved_width)  # Width percentage for right sidebar
    notes_sidebar_left_collapsed: reactive[bool] = reactive(False)
    notes_sidebar_right_collapsed: reactive[bool] = reactive(False)
    conv_char_sidebar_left_collapsed: reactive[bool] = reactive(False)
    conv_char_sidebar_right_collapsed: reactive[bool] = reactive(False)
    evals_sidebar_collapsed: reactive[bool] = reactive(False) # Added for Evals tab
    media_active_view: reactive[Optional[str]] = reactive(None)  # Added for Media tab navigation

    # Reactive variables for selected note details
    current_selected_note_id: reactive[Optional[str]] = reactive(None)
    current_selected_note_version: reactive[Optional[int]] = reactive(None)
    current_selected_note_title: reactive[Optional[str]] = reactive(None)
    current_selected_note_content: reactive[Optional[str]] = reactive("")
    
    # Notes tab UI state
    notes_unsaved_changes: reactive[bool] = reactive(False)
    notes_sort_by: reactive[str] = reactive("date_created")  # date_created, date_modified, title
    notes_sort_ascending: reactive[bool] = reactive(False)  # False = newest first
    notes_preview_mode: reactive[bool] = reactive(False)  # False = edit mode, True = preview mode
    
    # Auto-save related reactive variables
    notes_auto_save_enabled: reactive[bool] = reactive(True)  # Auto-save enabled by default
    notes_auto_save_timer: reactive[Optional[Timer]] = reactive(None)  # Timer reference for auto-save
    notes_last_save_time: reactive[Optional[float]] = reactive(None)  # Timestamp of last save
    notes_auto_save_status: reactive[str] = reactive("")  # Status: "", "saving", "saved"

    # --- Reactives for chat sidebar prompt display ---
    chat_sidebar_selected_prompt_id: reactive[Optional[int]] = reactive(None)
    chat_sidebar_selected_prompt_system: reactive[Optional[str]] = reactive(None)
    chat_sidebar_selected_prompt_user: reactive[Optional[str]] = reactive(None)

    # Chats
    current_chat_is_ephemeral: reactive[bool] = reactive(True)  # Start new chats as ephemeral
    # Reactive variable for current chat conversation ID
    current_chat_conversation_id: reactive[Optional[str]] = reactive(None)
    # Reactive variable for current conversation loaded in the Conversations, Characters & Prompts tab
    current_conv_char_tab_conversation_id: reactive[Optional[str]] = reactive(None)
    current_chat_active_character_data: reactive[Optional[Dict[str, Any]]] = reactive(None)
    current_ccp_character_details: reactive[Optional[Dict[str, Any]]] = reactive(None)
    current_ccp_character_image: Optional[Image.Image] = None
    
    # Chat Tabs Management (when enable_tabs is True)
    active_chat_tab_id: reactive[Optional[str]] = reactive(None)
    chat_sessions: reactive[Dict[str, Dict[str, Any]]] = reactive({})  # tab_id -> session_data dict

    # For Chat Sidebar Prompts section
    chat_sidebar_loaded_prompt_id: reactive[Optional[Union[int, str]]] = reactive(None)
    chat_sidebar_loaded_prompt_title_text: reactive[str] = reactive("")
    chat_sidebar_loaded_prompt_system_text: reactive[str] = reactive("")
    chat_sidebar_loaded_prompt_user_text: reactive[str] = reactive("")
    chat_sidebar_loaded_prompt_keywords_text: reactive[str] = reactive("")
    chat_sidebar_prompt_display_visible: reactive[bool] = reactive(False, layout=True)

    # Prompts
    current_prompt_id: reactive[Optional[int]] = reactive(None)
    current_prompt_uuid: reactive[Optional[str]] = reactive(None)
    current_prompt_name: reactive[Optional[str]] = reactive(None)
    current_prompt_author: reactive[Optional[str]] = reactive(None)
    current_prompt_details: reactive[Optional[str]] = reactive(None)
    current_prompt_system: reactive[Optional[str]] = reactive(None)
    current_prompt_user: reactive[Optional[str]] = reactive(None)
    current_prompt_keywords_str: reactive[Optional[str]] = reactive("") # Store as comma-sep string for UI
    current_prompt_version: reactive[Optional[int]] = reactive(None) # If DB provides it and you need it
    # is_new_prompt can be inferred from current_prompt_id being None

    # Media Tab
    _media_types_for_ui: List[str] = []
    _initial_media_view_slug: Optional[str] = reactive(slugify("All Media"))  # Default to "All Media" slug

    current_media_type_filter_slug: reactive[Optional[str]] = reactive(slugify("All Media"))  # Slug for filtering
    current_media_type_filter_display_name: reactive[Optional[str]] = reactive("All Media")  # Display name
    media_current_page: reactive[int] = reactive(1) # Search results pagination

    # current_media_search_term: reactive[str] = reactive("") # Handled by inputs directly
    current_loaded_media_item: reactive[Optional[Dict[str, Any]]] = reactive(None)
    _media_search_timers: Dict[str, Timer] = {}  # For debouncing per media type
    _media_sidebar_search_timer: Optional[Timer] = None # For chat sidebar media search debouncing

    # Add media_types_for_ui to store fetched types
    media_types_for_ui: List[str] = []
    _initial_media_view: Optional[str] = "media-view-video-audio"  # Default to the first sub-tab
    media_db: Optional[MediaDatabase] = None
    current_sidebar_media_item: Optional[Dict[str, Any]] = None # For chat sidebar media review

    # Settings mode for chat sidebar
    chat_settings_mode: reactive[str] = reactive("basic")  # "basic" or "advanced"
    chat_settings_search_query: reactive[str] = reactive("")  # Search query for settings

    # Search Tab's active sub-view reactives
    search_active_sub_tab: reactive[Optional[str]] = reactive(None)
    _initial_search_sub_tab_view: Optional[str] = SEARCH_VIEW_RAG_QA

    # Ingest Tab
    ingest_active_view: reactive[Optional[str]] = reactive("ingest-view-prompts")
    _initial_ingest_view: Optional[str] = "ingest-view-prompts"
    selected_prompt_files_for_import: List[Path] = []
    parsed_prompts_for_preview: List[Dict[str, Any]] = []
    last_prompt_import_dir: Optional[Path] = None
    selected_note_files_for_import: List[Path] = []
    parsed_notes_for_preview: List[Dict[str, Any]] = []
    last_note_import_dir: Optional[Path] = None
    # Add attributes to hold the handlers (optional, but can be useful)
    prompt_import_success_handler: Optional[Callable] = None
    prompt_import_failure_handler: Optional[Callable] = None
    character_import_success_handler: Optional[Callable] = None
    character_import_failure_handler: Optional[Callable] = None
    note_import_success_handler: Optional[Callable] = None
    note_import_failure_handler: Optional[Callable] = None

    # Tools Tab
    tools_settings_active_view: reactive[Optional[str]] = reactive("ts-view-general-settings")  # Default to general settings
    _initial_tools_settings_view: Optional[str] = "ts-view-general-settings"

    _prompt_search_timer: Optional[Timer] = None

    # LLM Inference Tab
    llm_active_view: reactive[Optional[str]] = reactive(None)
    _initial_llm_view: Optional[str] = "llm-view-llama-cpp"
    
    llamacpp_server_process: Optional[subprocess.Popen] = None
    llamafile_server_process: Optional[subprocess.Popen] = None
    vllm_server_process: Optional[subprocess.Popen] = None
    ollama_server_process: Optional[subprocess.Popen] = None
    mlx_server_process: Optional[subprocess.Popen] = None
    onnx_server_process: Optional[subprocess.Popen] = None

    # De-Bouncers
    _conv_char_search_timer: Optional[Timer] = None
    _conversation_search_timer: Optional[Timer] = None
    _notes_search_timer: Optional[Timer] = None
    _chat_sidebar_prompt_search_timer: Optional[Timer] = None # New timer
    
    # Flag to track if character filter has been populated
    _chat_character_filter_populated: bool = False

    # Make API_IMPORTS_SUCCESSFUL accessible if needed by old methods or directly
    API_IMPORTS_SUCCESSFUL = API_IMPORTS_SUCCESSFUL

    # User ID for notes, will be initialized in __init__
    current_user_id: str = "default_user" # Will be overridden by self.notes_user_id

    # For Chat Tab's Notes section
    current_chat_note_id: Optional[str] = None
    current_chat_note_version: Optional[int] = None

    # Shared state for tldw API requests
    _last_tldw_api_request_context: Dict[str, Any] = {}

    def __init__(self):
        # Track startup timing
        self._startup_start_time = time.perf_counter()
        self._startup_phases = {}
        
        # Tab switching optimization
        self._tab_switch_timer = None
        self._pending_tab_switch = None
        self._initialized_tabs = set()  # Track which tabs have been initialized
        
        # Reduce logging in production
        if not os.environ.get("TLDW_DEBUG"):
            logging.getLogger().setLevel(logging.INFO)  # Reduce to INFO level in production
            # Disable debug logging for performance
            logging.getLogger("tldw_chatbook").setLevel(logging.INFO)
        
        # Log initial memory usage only in debug mode
        if os.environ.get("TLDW_DEBUG"):
            log_resource_usage()
        log_counter("app_startup_initiated", 1, documentation="Application startup initiated")
        
        super().__init__()
        
        # Phase 1: Basic initialization
        phase_start = time.perf_counter()
        self.MediaDatabase = MediaDatabase
        self.app_config = load_settings()
        self.loguru_logger = loguru_logger
        self.loguru_logger.info(f"Loaded app_config - strip_thinking_tags: {self.app_config.get('chat_defaults', {}).get('strip_thinking_tags', 'NOT SET')}") # Make loguru_logger an instance variable for handlers
        self.prompts_client_id = "tldw_tui_client_v1" # Store client ID for prompts service
        self.db_status_manager = DBStatusManager(self)  # Initialize database status manager
        self._startup_phases["basic_init"] = time.perf_counter() - phase_start
        log_histogram("app_startup_phase_duration_seconds", self._startup_phases["basic_init"], 
                     labels={"phase": "basic_init"}, 
                     documentation="Duration of startup phase in seconds")
        

        # Phase 2: Attribute initialization
        phase_start = time.perf_counter()
        self.parsed_prompts_for_preview = [] # <<< INITIALIZATION for prompts
        self.last_prompt_import_dir = None

        self.selected_character_files_for_import = []
        self.parsed_characters_for_preview = [] # <<< INITIALIZATION for characters
        self.last_character_import_dir = None
        # Initialize Ingest Tab related attributes
        self.selected_prompt_files_for_import = []
        self.parsed_prompts_for_preview = []
        self.last_prompt_import_dir = Path.home()  # Or Path(".")
        self.selected_notes_files_for_import = []
        self.parsed_notes_for_preview = [] # <<< INITIALIZATION for notes
        self.last_notes_import_dir = None
        # Llama.cpp server process
        self.llamacpp_server_process = None
        # LlamaFile server process
        self.llamafile_server_process = None
        # vLLM server process
        self.vllm_server_process = None
        self.ollama_server_process = None
        self.mlx_server_process = None
        self.onnx_server_process = None
        self.media_current_page = 1
        self.media_search_current_page = 1
        self.media_search_total_pages = 1
        self._startup_phases["attribute_init"] = time.perf_counter() - phase_start
        log_histogram("app_startup_phase_duration_seconds", self._startup_phases["attribute_init"], 
                     labels={"phase": "attribute_init"}, 
                     documentation="Duration of startup phase in seconds")

        # Phase 3: Parallel initialization of independent services
        phase_start = time.perf_counter()
        
        # Prepare shared data
        user_name_for_notes = settings.get("USERS_NAME", "default_tui_user")
        self.notes_user_id = user_name_for_notes
        
        # Run independent initializations in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all independent initialization tasks
            futures = {
                executor.submit(self._init_notes_service, user_name_for_notes): "notes_service",
                executor.submit(self._init_providers_models): "providers_models",
                executor.submit(self._init_prompts_service): "prompts_service",
                executor.submit(self._init_media_db): "media_db"
            }
            
            # Wait for all tasks to complete and log individual timings
            for future in concurrent.futures.as_completed(futures):
                task_name = futures[future]
                try:
                    task_start = time.perf_counter()
                    result = future.result()
                    task_duration = time.perf_counter() - task_start
                    logger.info(f"Parallel init task '{task_name}' completed in {task_duration:.3f}s")
                except Exception as e:
                    logger.error(f"Parallel init task '{task_name}' failed: {e}", exc_info=True)
        
        # Log total parallel phase time
        parallel_duration = time.perf_counter() - phase_start
        self._startup_phases["parallel_init"] = parallel_duration
        log_histogram("app_startup_phase_duration_seconds", parallel_duration, 
                     labels={"phase": "parallel_init"}, 
                     documentation="Duration of parallel initialization phase")
        log_resource_usage()  # Check memory after parallel init

        # Providers, prompts, and media DB are initialized in parallel above
        # Just ensure we have defaults if parallel init failed
        if not hasattr(self, 'providers_models'):
            self.providers_models = {}

        # --- Initial Tab ---
        initial_tab_from_config = get_cli_setting("general", "default_tab", TAB_CHAT)
        self._initial_tab_value = initial_tab_from_config if initial_tab_from_config in ALL_TABS else TAB_CHAT
        if self._initial_tab_value != initial_tab_from_config: # Log if fallback occurred
            logging.warning(f"Default tab '{initial_tab_from_config}' from config not valid. Falling back to '{self._initial_tab_value}'.")
        logging.info(f"App __init__: Determined initial tab value: {self._initial_tab_value}")
        # current_tab reactive will be set in on_mount after UI is composed

        self._rich_log_handler: Optional[RichLogHandler] = None # For the RichLog widget in Logs tab

        # Prompts service is initialized in parallel above
        # Set up timer
        self._prompt_search_timer = None

        # Media DB is initialized in parallel above
        # Ensure we have media types for UI
        if not hasattr(self, '_media_types_for_ui'):
            self._media_types_for_ui = ["Error: Media DB not loaded"]

        self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self._media_types_for_ui VALUE: {self._media_types_for_ui}")
        self.loguru_logger.debug(f"ULTRA EARLY APP INIT: self._media_types_for_ui TYPE: {type(self._media_types_for_ui)}")

        # --- Setup Default view for CCP tab ---
        # Initialize self.ccp_active_view based on initial tab or default state if needed
        if self._initial_tab_value == TAB_CCP:
            self.ccp_active_view = "conversation_details_view"  # Default view for CCP tab
        # else: it will default to "conversation_details_view" anyway
        self._ui_ready = False  # Track if UI is fully composed
        self._shutting_down = False  # Track if app is shutting down

        # --- Assign DB instances for event handlers ---
        if self.prompts_service_initialized:
            # Get the database instance using the get_db_instance() function
            try:
                self.prompts_db = prompts_interop.get_db_instance()
                logging.info("Assigned prompts_interop.get_db_instance() to self.prompts_db")
            except RuntimeError as e:
                logging.error(f"Error getting prompts_db instance: {e}")
                self.prompts_db = None # Explicitly set to None
        else:
            self.prompts_db = None # Ensure it's None if service failed
            logging.warning("Prompts service not initialized, self.prompts_db set to None.")

        if self.notes_service and hasattr(self.notes_service, 'db') and self.notes_service.db:
            self.chachanotes_db = self.notes_service.db # ChaChaNotesDB is used by NotesInteropService
            logging.info("Assigned self.notes_service.db to self.chachanotes_db")
        else: # Fallback to global if notes_service didn't set it up as expected on itself
            lazy_db = get_chachanotes_db_lazy()
            if lazy_db:
                self.chachanotes_db = lazy_db
                logging.info("Assigned lazy-loaded chachanotes_db to self.chachanotes_db as fallback.")
            else:
                logging.error("ChaChaNotesDB (CharactersRAGDB) instance not found/assigned in app.__init__.")
                self.chachanotes_db = None # Explicitly set to None

        # --- Create the master handler map ---
        # This one-time setup makes the dispatcher clean and fast.
        self.button_handler_map = self._build_handler_map()
        
        # --- Initialize worker handler registry ---
        self._init_worker_handlers()
        
        # Log total initialization time
        total_init_time = time.perf_counter() - self._startup_start_time
        self._startup_phases["total_init"] = total_init_time
        log_histogram("app_startup_total_duration_seconds", total_init_time, 
                     documentation="Total application initialization time in seconds")
        
        # Log startup summary
        logger.info(f"=== STARTUP TIMING SUMMARY ===")
        logger.info(f"Total initialization time: {total_init_time:.3f} seconds")
        for phase, duration in self._startup_phases.items():
            if phase != "total_init":
                percentage = (duration / total_init_time) * 100 if total_init_time > 0 else 0
                logger.info(f"  {phase}: {duration:.3f}s ({percentage:.1f}%)")
        logger.info(f"==============================")
        
        # Final memory check
        log_resource_usage()


    def _init_notes_service(self, user_name_for_notes: str) -> None:
        """Initialize notes service - for parallel execution."""
        try:
            # Get the full path to the unified ChaChaNotes DB FILE
            chachanotes_db_file_path = get_chachanotes_db_path()
            logger.info(f"Unified ChaChaNotes DB file path: {chachanotes_db_file_path}")
            
            # Determine the PARENT DIRECTORY for NotesInteropService's 'base_db_directory'
            actual_base_directory_for_service = chachanotes_db_file_path.parent
            logger.info(f"Notes for user '{user_name_for_notes}' will use the unified DB: {chachanotes_db_file_path}")
            
            self.notes_service = NotesInteropService(
                base_db_directory=actual_base_directory_for_service,
                api_client_id="tldw_tui_client_v1",
                global_db_to_use=get_chachanotes_db_lazy()
            )
            logger.info(f"NotesInteropService successfully initialized for user '{user_name_for_notes}'.")
        except Exception as e:
            logger.error(f"Failed to initialize NotesInteropService: {e}", exc_info=True)
            self.notes_service = None
    
    def _init_providers_models(self) -> None:
        """Initialize providers and models - for parallel execution."""
        try:
            self.providers_models = get_cli_providers_and_models()
            logger.info(f"Successfully retrieved providers_models. Count: {len(self.providers_models)}. Keys: {list(self.providers_models.keys())}")
        except Exception as e:
            logger.error(f"Failed to get providers and models: {e}", exc_info=True)
            self.providers_models = {}
    
    def _init_prompts_service(self) -> None:
        """Initialize prompts service - for parallel execution."""
        self.prompts_service_initialized = False
        try:
            prompts_db_path = get_prompts_db_path()
            prompts_db_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_interop.initialize_interop(db_path=prompts_db_path, client_id=self.prompts_client_id)
            self.prompts_service_initialized = True
            logger.info(f"Prompts Interop Service initialized with DB: {prompts_db_path}")
        except Exception as e:
            self.prompts_service_initialized = False
            logger.error(f"Failed to initialize Prompts Interop Service: {e}", exc_info=True)
    
    def _init_media_db(self) -> None:
        """Initialize media database - for parallel execution."""
        try:
            media_db_path = get_media_db_path()
            # Get integrity check configuration
            check_integrity = self.app_config.get('database', {}).get('check_integrity_on_startup', False)
            self.media_db = MediaDatabase(
                db_path=media_db_path, 
                client_id=CLI_APP_CLIENT_ID,
                check_integrity_on_startup=check_integrity
            )
            logger.info(f"Media_DB_v2 initialized successfully for client '{CLI_APP_CLIENT_ID}' at {media_db_path}")
            
            # Pre-fetch media types for UI
            if self.media_db:
                db_types = self.media_db.get_distinct_media_types(include_deleted=False, include_trash=False)
                self._media_types_for_ui = ["All Media"] + sorted(list(set(db_types)))
                logger.info(f"Pre-fetched {len(self._media_types_for_ui)} media types for UI.")
            else:
                self._media_types_for_ui = ["Error: Media DB not loaded"]
        except Exception as e:
            logger.error(f"Failed to initialize media DB: {e}", exc_info=True)
            self.media_db = None
            self._media_types_for_ui = ["Error: Exception fetching media types"]
    
    def _init_worker_handlers(self) -> None:
        """Initialize the worker handler registry and register all handlers."""
        self.worker_handler_registry = WorkerHandlerRegistry(self)
        
        # Register all worker handlers
        self.worker_handler_registry.register(ChatWorkerHandler(self))
        self.worker_handler_registry.register(ServerWorkerHandler(self))
        self.worker_handler_registry.register(AIGenerationHandler(self))
        self.worker_handler_registry.register(MiscWorkerHandler(self))
        
        self.loguru_logger.info("Worker handler registry initialized with all handlers")

    def _build_handler_map(self) -> dict:
        """Constructs the master button handler map from all event modules."""

        # --- Generic, Awaitable Helper Handlers ---
        async def _handle_nav(app: 'TldwCli', event: Button.Pressed, *, prefix: str, reactive_attr: str) -> None:
            """Generic handler for switching views within a tab."""
            view_to_activate = event.button.id.replace(f"{prefix}-nav-", f"{prefix}-view-")
            app.loguru_logger.info(f"_handle_nav called: Nav button '{event.button.id}' pressed. Prefix: '{prefix}', Reactive attr: '{reactive_attr}', Activating view '{view_to_activate}'.")
            old_value = getattr(app, reactive_attr, None)
            setattr(app, reactive_attr, view_to_activate)
            new_value = getattr(app, reactive_attr, None)
            app.loguru_logger.info(f"_handle_nav: Set {reactive_attr} from '{old_value}' to '{new_value}'")

        async def _handle_sidebar_toggle(app: 'TldwCli', event: Button.Pressed, *, reactive_attr: str) -> None:
            """Generic handler for toggling a sidebar's collapsed state."""
            setattr(app, reactive_attr, not getattr(app, reactive_attr))

        # --- LLM Management Handlers ---
        llm_handlers_map = {
            **llm_management_events.LLM_MANAGEMENT_BUTTON_HANDLERS,
            **llm_nav_events.LLM_NAV_BUTTON_HANDLERS,
            **llm_management_events_mlx_lm.MLX_LM_BUTTON_HANDLERS,
            **llm_management_events_ollama.OLLAMA_BUTTON_HANDLERS,
            **llm_management_events_onnx.ONNX_BUTTON_HANDLERS,
            **llm_management_events_transformers.TRANSFORMERS_BUTTON_HANDLERS,
            **llm_management_events_vllm.VLLM_BUTTON_HANDLERS,
        }

        # --- Chat Handlers ---

        chat_handlers_map = {
            **chat_events.CHAT_BUTTON_HANDLERS,
            **chat_events_sidebar.CHAT_SIDEBAR_BUTTON_HANDLERS,
            "toggle-chat-left-sidebar": functools.partial(_handle_sidebar_toggle, reactive_attr="chat_sidebar_collapsed"),
            "toggle-chat-right-sidebar": functools.partial(_handle_sidebar_toggle, reactive_attr="chat_right_sidebar_collapsed"),
        }

        # --- Media Tab Handlers (NEW DYNAMIC WAY) ---
        media_handlers_map = {}
        for media_type_name in self._media_types_for_ui:
            slug = slugify(media_type_name)
            media_handlers_map[f"media-nav-{slug}"] = media_events.handle_media_nav_button_pressed
            media_handlers_map[f"media-load-selected-button-{slug}"] = media_events.handle_media_load_selected_button_pressed
            media_handlers_map[f"media-prev-page-button-{slug}"] = media_events.handle_media_page_change_button_pressed
            media_handlers_map[f"media-next-page-button-{slug}"] = media_events.handle_media_page_change_button_pressed
        
        # Add handlers for special media sub-tabs
        media_handlers_map["media-nav-analysis-review"] = media_events.handle_media_nav_button_pressed
        media_handlers_map["media-nav-collections-tags"] = media_events.handle_media_nav_button_pressed
        media_handlers_map["media-nav-multi-item-review"] = media_events.handle_media_nav_button_pressed

        # --- Search Handlers ---
        search_handlers = {
            SEARCH_NAV_RAG_QA: functools.partial(_handle_nav, prefix="search", reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_RAG_CHAT: functools.partial(_handle_nav, prefix="search", reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_RAG_MANAGEMENT: functools.partial(_handle_nav, prefix="search",
                                                         reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_WEB_SEARCH: functools.partial(_handle_nav, prefix="search",
                                                     reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_EMBEDDINGS_CREATE: functools.partial(_handle_nav, prefix="search", 
                                                           reactive_attr="search_active_sub_tab"),
            SEARCH_NAV_EMBEDDINGS_MANAGE: functools.partial(_handle_nav, prefix="search",
                                                           reactive_attr="search_active_sub_tab"),
            **search_events.SEARCH_BUTTON_HANDLERS,
        }

        # --- Ingest Handlers ---
        ingest_handlers_map = {
            **ingest_events.INGEST_BUTTON_HANDLERS,
            # Add nav handlers using the helper
            **{button_id: functools.partial(_handle_nav, prefix="ingest", reactive_attr="ingest_active_view")
               for button_id in INGEST_NAV_BUTTON_IDS}
        }

        # --- Tools & Settings Handlers ---
        tools_settings_handlers = {
            "ts-nav-general-settings": functools.partial(_handle_nav, prefix="ts",
                                                         reactive_attr="tools_settings_active_view"),
            "ts-nav-config-file-settings": functools.partial(_handle_nav, prefix="ts",
                                                             reactive_attr="tools_settings_active_view"),
            "ts-nav-db-tools": functools.partial(_handle_nav, prefix="ts",
                                                 reactive_attr="tools_settings_active_view"),
            "ts-nav-appearance": functools.partial(_handle_nav, prefix="ts",
                                                   reactive_attr="tools_settings_active_view"),
        }

        # --- Evals Handler ---
        evals_handlers = {
            "toggle-evals-sidebar": functools.partial(_handle_sidebar_toggle, reactive_attr="evals_sidebar_collapsed"),
        }

        # Master map organized by tab
        return {
            TAB_CHAT: chat_handlers_map,
            TAB_CCP: {
                **ccp_handlers.CCP_BUTTON_HANDLERS,
                "toggle-conv-char-left-sidebar": functools.partial(_handle_sidebar_toggle,
                                                                   reactive_attr="conv_char_sidebar_left_collapsed"),
                "toggle-conv-char-right-sidebar": functools.partial(_handle_sidebar_toggle,
                                                                    reactive_attr="conv_char_sidebar_right_collapsed"),
            },
            TAB_NOTES: {
                **notes_events.NOTES_BUTTON_HANDLERS,
                "toggle-notes-sidebar-left": functools.partial(_handle_sidebar_toggle,
                                                               reactive_attr="notes_sidebar_left_collapsed"),
                "toggle-notes-sidebar-right": functools.partial(_handle_sidebar_toggle,
                                                                reactive_attr="notes_sidebar_right_collapsed"),
            },
            TAB_MEDIA: {
                **media_events.MEDIA_BUTTON_HANDLERS,
                **{f"media-nav-{slugify(media_type)}": functools.partial(_handle_nav, prefix="media",
                                                                               reactive_attr="media_active_view")
                   for media_type in self._media_types_for_ui},
                "media-nav-all-media": functools.partial(_handle_nav, prefix="media",
                                                         reactive_attr="media_active_view"),
            },
            TAB_INGEST: ingest_handlers_map,
            TAB_LLM: llm_handlers_map,
            TAB_LOGS: app_lifecycle.APP_LIFECYCLE_BUTTON_HANDLERS,
            TAB_TOOLS_SETTINGS: tools_settings_handlers,
            TAB_CUSTOMIZE: {},  # Customize handles its own events
            TAB_SEARCH: search_handlers,
            TAB_EVALS: evals_handlers,
            TAB_CODING: {},  # Empty for now - coding handles its own events
            TAB_STTS: {}, # STTS handles its own events
            TAB_STUDY: {}, # Study handles its own events
            TAB_SUBSCRIPTIONS: {
                "subscription-add-button": subscription_events.handle_add_subscription,
                "subscription-check-all-button": subscription_events.handle_check_all_subscriptions,
                "subscription-accept-button": subscription_events.handle_subscription_item_action,
                "subscription-ignore-button": subscription_events.handle_subscription_item_action,
                "subscription-mark-reviewed-button": subscription_events.handle_subscription_item_action,
            },
        }

    def _setup_logging(self):
        """Set up logging for the application.

        If early logging was already initialized, this will just set up the RichLogHandler
        for the UI log display widget.
        """
        # Check if we're running as a module (via entry point) or as a script
        if hasattr(self, '_early_logging_initialized') and self._early_logging_initialized:
            # Early logging was already initialized, just set up the RichLogHandler
            logging.info("Logging already initialized early, setting up UI log handlers only")
            try:
                log_display_widget = self.query_one("#app-log-display", RichLog)
                if not self._rich_log_handler:
                    self._rich_log_handler = RichLogHandler(log_display_widget)
                    rich_log_handler_level_str = self.app_config.get("logging", {}).get("rich_log_level", "DEBUG").upper()
                    rich_log_handler_level = getattr(logging, rich_log_handler_level_str, logging.DEBUG)
                    self._rich_log_handler.setLevel(rich_log_handler_level)
                    logging.getLogger().addHandler(self._rich_log_handler)
                    logging.info(f"Added RichLogHandler to existing logging setup (Level: {logging.getLevelName(self._rich_log_handler.level)}).")
            except QueryError:
                logging.error("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
            except Exception as e:
                logging.error(f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True)
        else:
            # No early logging, do full initialization
            configure_application_logging(self)

    def compose(self) -> ComposeResult:
        compose_start = time.perf_counter()
        self._ui_compose_start_time = compose_start  # Store for later reference
        logging.debug("App composing UI...")
        log_counter("ui_compose_started", 1, documentation="UI composition started")
        
        # Check if splash screen is enabled
        splash_enabled = get_cli_setting("splash_screen", "enabled", True)
        logging.info(f"Splash screen enabled: {splash_enabled}")
        if splash_enabled:
            # Get splash screen configuration
            splash_duration = get_cli_setting("splash_screen", "duration", 1.5)
            splash_skip = get_cli_setting("splash_screen", "skip_on_keypress", True)
            splash_progress = get_cli_setting("splash_screen", "show_progress", True)
            splash_card = get_cli_setting("splash_screen", "card_selection", "random")
            logging.info(f"Creating splash screen - duration: {splash_duration}, card: {splash_card}")
            
            # Create and yield splash screen
            self._splash_screen_widget = SplashScreen(
                card_name=splash_card if splash_card != "random" else None,
                duration=splash_duration,
                skip_on_keypress=splash_skip,
                show_progress=splash_progress,
                id="app-splash-screen"
            )
            self.splash_screen_active = True
            yield self._splash_screen_widget
            logging.info("Splash screen yielded, returning early from compose")
            
            # Important: Return early to only show splash screen initially
            # The main UI will be mounted after splash screen is closed
            return
        
        # If splash screen is disabled, compose the main UI immediately
        yield from self._compose_main_ui()
    
    def _compose_main_ui(self) -> ComposeResult:
        """Compose the main UI by yielding created widgets."""
        widgets = self._create_main_ui_widgets()
        for widget in widgets:
            yield widget
        
    def _create_main_ui_widgets(self) -> List[Widget]:
        """Create the main UI widgets (called after splash screen or immediately if disabled)."""
        widgets = []
        compose_phases = {}  # Track timing for each phase
        
        # Phase: Title Bar
        phase_start = time.perf_counter()
        component_start = time.perf_counter()
        widgets.append(TitleBar())
        titlebar_time = time.perf_counter() - component_start
        compose_phases["titlebar"] = titlebar_time
        log_histogram("app_component_creation_duration_seconds", titlebar_time,
                     labels={"component": "titlebar"}, 
                     documentation="Time to create UI component")
        logger.info(f"TitleBar created in {titlebar_time:.3f}s")

        # Phase: Navigation (TabBar, TabLinks, or TabDropdown)
        phase_start = time.perf_counter()
        use_dropdown = get_cli_setting("general", "use_dropdown_navigation", False)
        use_links = get_cli_setting("general", "use_link_navigation", True)  # Default to links
        component_start = time.perf_counter()
        
        if use_dropdown:
            # Use dropdown navigation
            widgets.append(TabDropdown(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value))
            logger.info("Using dropdown navigation for tabs")
        elif use_links:
            # Use single-line link navigation
            widgets.append(TabLinks(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value))
            logger.info("Using single-line link navigation for tabs")
        else:
            # Use traditional tab bar
            widgets.append(TabBar(tab_ids=ALL_TABS, initial_active_tab=self._initial_tab_value))
            logger.info("Using traditional tab bar navigation")
        
        nav_time = time.perf_counter() - component_start
        compose_phases["navigation"] = nav_time
        log_histogram("app_component_creation_duration_seconds", nav_time,
                     labels={"component": "navigation"}, 
                     documentation="Time to create UI component")
        logger.info(f"Navigation created in {nav_time:.3f}s")

        # Content area - all windows
        content_area_start = time.perf_counter()
        
        # Check config for which chat window to use
        use_enhanced_chat = get_cli_setting("chat_defaults", "use_enhanced_window", False)
        chat_window_class = ChatWindowEnhanced if use_enhanced_chat else ChatWindow
        logger.info(f"Using {'enhanced' if use_enhanced_chat else 'basic'} chat window (use_enhanced_window={use_enhanced_chat})")
        
        # Create content container with all windows
        content_container = Container(id="content")
        
        windows = [
            ("chat", chat_window_class, "chat-window"),
            ("ccp", CCPWindow, "conversations_characters_prompts-window"),
            ("notes", NotesWindow, "notes-window"),
            ("media", MediaWindow, "media-window"),
            ("search", SearchWindow, "search-window"),
            ("ingest", NewIngestWindow, "ingest-window"),
            ("tools_settings", ToolsSettingsWindow, "tools_settings-window"),
            ("llm_management", LLMManagementWindow, "llm_management-window"),
            ("customize", CustomizeWindow, "customize-window"),
            ("logs", LogsWindow, "logs-window"),
            ("coding", CodingWindow, "coding-window"),
            ("stats", StatsWindow, "stats-window"),
            ("evals", EvalsWindow, "evals-window"),
            ("stts", STTSWindow, "stts-window"),
            ("study", StudyWindow, "study-window"),
            ("chatbooks", ChatbooksWindow, "chatbooks-window"),
        ]
        
        # Add subscriptions tab if available
        if SUBSCRIPTIONS_AVAILABLE and SubscriptionWindow:
            windows.append(("subscriptions", SubscriptionWindow, "subscriptions-window"))
        
        # Create window widgets and compose them into the container properly
        initial_tab = self._initial_tab_value
        window_creation_times = {}  # Track individual window creation times
        
        for window_name, window_class, window_id in windows:
            window_start = time.perf_counter()
            is_initial_window = window_id == f"{initial_tab}-window"
            
            # Always load LogsWindow immediately to capture logs, and the initial window
            if is_initial_window or window_id == "logs-window":
                # Create the actual window for the initial tab and logs tab
                logger.info(f"Creating actual window for {window_name}")
                creation_start = time.perf_counter()
                window_widget = window_class(self, id=window_id, classes="window")
                creation_time = time.perf_counter() - creation_start
                window_creation_times[window_name] = creation_time
                logger.info(f"Window {window_name} created in {creation_time:.3f}s")
                # For non-initial windows, make them invisible initially
                if not is_initial_window:
                    window_widget.display = False
            else:
                # Create a placeholder for other tabs
                logger.debug(f"Creating placeholder for tab: {window_name}")
                window_widget = PlaceholderWindow(self, window_class, window_id, classes="window")
                window_widget.display = False  # Hide placeholder initially
                window_creation_times[window_name] = time.perf_counter() - window_start
            
            # Mount the window widget into the container
            mount_start = time.perf_counter()
            content_container._add_child(window_widget)
            mount_time = time.perf_counter() - mount_start
            if mount_time > 0.01:  # Log if mounting takes more than 10ms
                logger.warning(f"Mounting {window_name} took {mount_time:.3f}s")
        
        # Log window creation summary
        if window_creation_times:
            sorted_times = sorted(window_creation_times.items(), key=lambda x: x[1], reverse=True)
            logger.info("=== WINDOW CREATION TIMES ===")
            for window_name, creation_time in sorted_times[:5]:  # Top 5 slowest
                logger.info(f"  {window_name}: {creation_time:.3f}s")
            logger.info("=============================")
        
        # Store for later analysis
        self._window_creation_times = window_creation_times
        
        widgets.append(content_container)
        
        log_histogram("app_component_creation_duration_seconds", time.perf_counter() - content_area_start,
                     labels={"component": "content_area_all_windows"}, 
                     documentation="Time to create UI component")

        # Yield the new AppFooterStatus widget instead of the old Footer
        component_start = time.perf_counter()
        widgets.append(AppFooterStatus(id="app-footer-status"))
        log_histogram("app_component_creation_duration_seconds", time.perf_counter() - component_start,
                     labels={"component": "footer"}, 
                     documentation="Time to create UI component")
        
        compose_duration = time.perf_counter() - self._ui_compose_start_time
        self._ui_compose_end_time = time.perf_counter()  # Store compose end time
        log_histogram("app_compose_duration_seconds", compose_duration,
                     documentation="Total time for compose() method")
        log_counter("ui_compose_completed", 1, documentation="UI composition completed")
        logging.debug(f"App compose finished in {compose_duration:.3f} seconds")
        log_resource_usage()  # Check memory after compose
        
        return widgets
    

    @on(ChatMessage.Action)
    async def handle_chat_message_action(self, event: ChatMessage.Action) -> None:
        """Handles actions (edit, copy, etc.) from within a ChatMessage widget."""
        button_classes = " ".join(event.button.classes) # Get class string for logging
        self.loguru_logger.debug(
            f"ChatMessage.Action received for button "
            f"(Classes: {button_classes}, Label: '{event.button.label}') "
            f"on message role: {event.message_widget.role}"
        )
        # The event directly gives us the context we need.
        # Now we call the existing handler function with the correct arguments.
        await chat_events.handle_chat_action_button_pressed(
            self, event.button, event.message_widget
        )

    @on(ChatMessageEnhanced.Action)
    async def handle_chat_message_enhanced_action(self, event: ChatMessageEnhanced.Action) -> None:
        """Handles actions (edit, copy, etc.) from within a ChatMessageEnhanced widget."""
        button_classes = " ".join(event.button.classes) # Get class string for logging
        self.loguru_logger.debug(
            f"ChatMessageEnhanced.Action received for button "
            f"(Classes: {button_classes}, Label: '{event.button.label}') "
            f"on message role: {event.message_widget.role}"
        )
        # The event directly gives us the context we need.
        # Now we call the existing handler function with the correct arguments.
        await chat_events.handle_chat_action_button_pressed(
            self, event.button, event.message_widget
        )

    @on(TTSRequestEvent)
    async def handle_tts_request_event(self, event: TTSRequestEvent) -> None:
        """Handle TTS generation request."""
        self.loguru_logger.info(f"TTS request received for text: '{event.text[:50]}...'")
        if self._tts_handler:
            await self._tts_handler.handle_tts_request(event)
        else:
            self.loguru_logger.error("TTS handler not initialized")
            await self.post_message(
                TTSCompleteEvent(
                    message_id=event.message_id or "unknown",
                    error="TTS service not available"
                )
            )

    @on(TTSCompleteEvent)
    async def handle_tts_complete_event(self, event: TTSCompleteEvent) -> None:
        """Handle TTS generation completion."""
        self.loguru_logger.info(f"TTS complete for message {event.message_id}")
        
        if event.error:
            self.notify(f"TTS failed: {event.error}", severity="error")
            # Update widget state back to idle on error
            try:
                if event.message_id:
                    # Find the message widget and update state
                    for message_widget in list(self.query(ChatMessage)) + list(self.query(ChatMessageEnhanced)):
                        if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                            # Update TTS state to idle on error
                            if hasattr(message_widget, 'update_tts_state'):
                                message_widget.update_tts_state("idle")
                            # Remove TTS generating class
                            text_widget = message_widget.query_one(".message-text", Markdown)
                            text_widget.remove_class("tts-generating")
                            break
            except Exception as e:
                self.loguru_logger.error(f"Error updating message UI: {e}")
        else:
            # Update widget state to ready with audio file
            if event.audio_file and event.audio_file.exists():
                try:
                    if event.message_id:
                        # Find the message widget and update state
                        for message_widget in list(self.query(ChatMessage)) + list(self.query(ChatMessageEnhanced)):
                            if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                                # Update TTS state to ready with audio file
                                if hasattr(message_widget, 'update_tts_state'):
                                    message_widget.update_tts_state("ready", event.audio_file)
                                # Remove TTS generating class
                                try:
                                    text_widget = message_widget.query_one(".message-text", Markdown)
                                    text_widget.remove_class("tts-generating")
                                except Exception:
                                    pass
                                break
                    # Don't automatically play or delete - let user control playback
                    self.notify("TTS audio ready - click play to listen", severity="information")
                except Exception as e:
                    self.loguru_logger.error(f"Error playing audio: {e}")
                    self.notify("Failed to play audio", severity="error")
            
            # Remove TTS generating class from message
            try:
                if event.message_id:
                    for message_widget in list(self.query(ChatMessage)) + list(self.query(ChatMessageEnhanced)):
                        if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                            text_widget = message_widget.query_one(".message-text", Markdown)
                            text_widget.remove_class("tts-generating")
                            break
            except Exception as e:
                self.loguru_logger.error(f"Error updating message UI: {e}")
    
    @on(TTSProgressEvent)
    async def handle_tts_progress_event(self, event: TTSProgressEvent) -> None:
        """Handle TTS generation progress updates."""
        self.loguru_logger.debug(f"TTS progress for message {event.message_id}: {event.progress:.0%} - {event.status}")
        
        try:
            if event.message_id:
                # Find the message widget and update progress
                for message_widget in self.query(ChatMessage).union(self.query(ChatMessageEnhanced)):
                    if getattr(message_widget, 'message_id_internal', None) == event.message_id:
                        # Update TTS progress
                        if hasattr(message_widget, 'update_tts_progress'):
                            message_widget.update_tts_progress(event.progress, event.status)
                        break
        except Exception as e:
            self.loguru_logger.error(f"Error updating TTS progress: {e}")

    @on(TTSPlaybackEvent)
    async def handle_tts_playback_event(self, event: TTSPlaybackEvent) -> None:
        """Handle TTS playback control."""
        if self._tts_handler:
            await self._tts_handler.handle_tts_playback(event)
    
    @on(STTSPlaygroundGenerateEvent)
    async def handle_stts_playground_generate_event(self, event: STTSPlaygroundGenerateEvent) -> None:
        """Handle S/TT/S playground generation request."""
        self.loguru_logger.info(f"S/TT/S generation request: provider={event.provider}, model={event.model}")
        if self._stts_handler:
            await self._stts_handler.handle_playground_generate(event)
        else:
            self.loguru_logger.error("S/TT/S handler not initialized")
            self.notify("S/TT/S service not available", severity="error")
    
    @on(STTSSettingsSaveEvent)
    async def handle_stts_settings_save_event(self, event: STTSSettingsSaveEvent) -> None:
        """Handle S/TT/S settings save."""
        if self._stts_handler:
            await self._stts_handler.handle_settings_save(event)
    
    @on(STTSAudioBookGenerateEvent)
    async def handle_stts_audiobook_generate_event(self, event: STTSAudioBookGenerateEvent) -> None:
        """Handle audiobook generation request."""
        if self._stts_handler:
            await self._stts_handler.handle_audiobook_generate(event)

    def switch_ccp_center_view(self, view_name: str) -> None:
        """Switch the center pane view in the CCP tab."""
        valid_views = {
            "conversations": "conversation_messages_view",
            "character": "character_card_view", 
            "character_editor": "character_editor_view",
            "prompt_editor": "prompt_editor_view",
            "dictionary": "dictionary_view",
            "dictionary_editor": "dictionary_editor_view"
        }
        
        if view_name not in valid_views:
            self.loguru_logger.warning(f"Invalid CCP view name: {view_name}")
            return
            
        # Map to the actual view name used by the watcher
        actual_view = valid_views[view_name]
        
        # Update the reactive which will trigger the watcher
        self.ccp_active_view = actual_view
        self.loguru_logger.info(f"Switched CCP center view to: {view_name}")

    # --- Watcher for CCP Active View ---
    def watch_ccp_active_view(self, old_view: Optional[str], new_view: str) -> None:
        loguru_logger.debug(f"CCP active view changing from '{old_view}' to: '{new_view}'")
        if not self._ui_ready:
            loguru_logger.debug("watch_ccp_active_view: UI not ready, returning.")
            return
        try:
            conversation_messages_view = self.query_one("#ccp-conversation-messages-view")
            prompt_editor_view = self.query_one("#ccp-prompt-editor-view")
            character_card_view = self.query_one("#ccp-character-card-view")
            character_editor_view = self.query_one("#ccp-character-editor-view")
            dictionary_view = self.query_one("#ccp-dictionary-view")
            dictionary_editor_view = self.query_one("#ccp-dictionary-editor-view")

            # Default all to hidden, then enable the correct one
            conversation_messages_view.display = False
            prompt_editor_view.display = False
            character_card_view.display = False
            character_editor_view.display = False
            dictionary_view.display = False
            dictionary_editor_view.display = False

            # REMOVE or COMMENT OUT the query for llm_settings_container_right:
            # llm_settings_container_right = self.query_one("#ccp-right-pane-llm-settings-container")
            # conv_details_collapsible_right = self.query_one("#ccp-conversation-details-collapsible", Collapsible) # Keep if you manipulate its collapsed state

            if new_view == "prompt_editor_view":
                # Center Pane: Show Prompt Editor
                prompt_editor_view.display = True
                # LLM settings container is gone, no need to hide it.
                # llm_settings_container_right.display = False

                # Optionally, manage collapsed state of other sidebars
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = False

                # Focus an element in prompt editor
                try:
                    self.query_one("#ccp-editor-prompt-name-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus prompt name input in editor view.")

            elif new_view == "character_editor_view":
                # Center Pane: Show Character Editor
                character_editor_view.display = True
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                loguru_logger.info("Character editor view activated. Focus pending specific input fields.")

            elif new_view == "character_card_view":
                # Center Pane: Show Character Card Display
                character_card_view.display = True
                character_editor_view.display = False
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                loguru_logger.info("Character card display view activated.")

                if self.current_ccp_character_details:
                    details = self.current_ccp_character_details
                    loguru_logger.info(f"Populating character card with details for: {details.get('name', 'Unknown')}")
                    try:
                        self.query_one("#ccp-card-name-display", Static).update(details.get("name", "N/A"))
                        self.query_one("#ccp-card-description-display", TextArea).text = details.get("description", "")
                        self.query_one("#ccp-card-personality-display", TextArea).text = details.get("personality", "")
                        self.query_one("#ccp-card-scenario-display", TextArea).text = details.get("scenario", "")
                        self.query_one("#ccp-card-first-message-display", TextArea).text = details.get("first_message", "")
                        
                        # Populate V2 Character Card fields
                        self.query_one("#ccp-card-creator-notes-display", TextArea).text = details.get("creator_notes") or ""
                        self.query_one("#ccp-card-system-prompt-display", TextArea).text = details.get("system_prompt") or ""
                        self.query_one("#ccp-card-post-history-instructions-display", TextArea).text = details.get("post_history_instructions") or ""
                        
                        # Handle alternate greetings (array to text)
                        alternate_greetings = details.get("alternate_greetings", [])
                        self.query_one("#ccp-card-alternate-greetings-display", TextArea).text = "\n".join(alternate_greetings) if alternate_greetings else ""
                        
                        # Handle tags (array to comma-separated)
                        tags = details.get("tags", [])
                        self.query_one("#ccp-card-tags-display", Static).update(", ".join(tags) if tags else "None")
                        
                        self.query_one("#ccp-card-creator-display", Static).update(details.get("creator") or "N/A")
                        self.query_one("#ccp-card-version-display", Static).update(details.get("character_version") or "N/A")
                        
                        # Handle keywords (array to comma-separated)
                        keywords = details.get("keywords", [])
                        self.query_one("#ccp-card-keywords-display", Static).update(", ".join(keywords) if keywords else "None")

                        image_placeholder = self.query_one("#ccp-card-image-placeholder", Static)
                        # Check if the character has an image in the database
                        if details.get("image"):
                            try:
                                # Convert image bytes to PIL Image for display
                                from PIL import Image
                                import io
                                import base64
                                
                                image_bytes = details["image"]
                                img = Image.open(io.BytesIO(image_bytes))
                                
                                # For now, just show image info until we implement proper image display
                                # In a real implementation, you'd convert to a displayable format
                                image_info = f"PNG Image: {img.width}x{img.height} pixels"
                                image_placeholder.update(image_info)
                                
                                # Store the PIL image for potential future use
                                self.current_ccp_character_image = img
                            except Exception as e:
                                loguru_logger.error(f"Error processing character image: {e}")
                                image_placeholder.update("Error loading image")
                        else:
                            image_placeholder.update("No image available")
                            self.current_ccp_character_image = None
                        loguru_logger.debug("Character card widgets populated.")
                    except QueryError as qe:
                        loguru_logger.error(f"QueryError populating character card: {qe}", exc_info=True)
                else:
                    loguru_logger.info("No character details available to populate card view.")
                    try:
                        self.query_one("#ccp-card-name-display", Static).update("N/A")
                        self.query_one("#ccp-card-description-display", TextArea).text = ""
                        self.query_one("#ccp-card-personality-display", TextArea).text = ""
                        self.query_one("#ccp-card-scenario-display", TextArea).text = ""
                        self.query_one("#ccp-card-first-message-display", TextArea).text = ""
                        
                        # Clear V2 Character Card fields
                        self.query_one("#ccp-card-creator-notes-display", TextArea).text = ""
                        self.query_one("#ccp-card-system-prompt-display", TextArea).text = ""
                        self.query_one("#ccp-card-post-history-instructions-display", TextArea).text = ""
                        self.query_one("#ccp-card-alternate-greetings-display", TextArea).text = ""
                        self.query_one("#ccp-card-tags-display", Static).update("None")
                        self.query_one("#ccp-card-creator-display", Static).update("N/A")
                        self.query_one("#ccp-card-version-display", Static).update("N/A")
                        self.query_one("#ccp-card-keywords-display", Static).update("None")
                        
                        self.query_one("#ccp-card-image-placeholder", Static).update("No character loaded")
                        self.current_ccp_character_image = None
                        loguru_logger.debug("Character card widgets cleared.")
                    except QueryError as qe:
                        loguru_logger.error(f"QueryError clearing character card: {qe}", exc_info=True)

            elif new_view == "dictionary_view":
                # Center Pane: Show Dictionary Display
                dictionary_view.display = True
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-dictionary-details-collapsible", Collapsible).collapsed = False
                loguru_logger.info("Dictionary display view activated.")

            elif new_view == "dictionary_editor_view":
                # Center Pane: Show Dictionary Editor
                dictionary_editor_view.display = True
                # Optionally manage right-pane collapsibles
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True
                self.query_one("#ccp-dictionary-details-collapsible", Collapsible).collapsed = False
                loguru_logger.info("Dictionary editor view activated.")
                # Focus on dictionary name input
                try:
                    self.query_one("#ccp-editor-dict-name-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus dictionary name input in editor view.")

            elif new_view == "conversation_details_view" or new_view == "conversation_messages_view":
                # Center Pane: Show Conversation Messages
                conversation_messages_view.display = True
                # LLM settings container is gone, no need to show it.
                # llm_settings_container_right.display = True
                self.query_one("#ccp-conversation-details-collapsible", Collapsible).collapsed = False
                self.query_one("#ccp-prompt-details-collapsible", Collapsible).collapsed = True

                try:
                    # If a conversation is loaded, maybe focus its title in right pane
                    if self.current_conv_char_tab_conversation_id:
                        self.query_one("#conv-char-title-input", Input).focus()
                    else:  # Otherwise, maybe focus the search in left pane
                        self.query_one("#conv-char-search-input", Input).focus()
                except QueryError:
                    loguru_logger.warning("Could not focus default element in conversation details view.")
            else:  # Default or unknown view (treat as conversation_details_view)
                # Center Pane: Show Conversation Messages (default)
                conversation_messages_view.display = True
                loguru_logger.warning(
                    f"Unknown ccp_active_view: {new_view}, defaulting to conversation_details_view.")

        except QueryError as e:
            loguru_logger.exception(f"UI component not found during CCP view switch: {e}")
        except Exception as e_watch:
            loguru_logger.exception(f"Unexpected error in watch_ccp_active_view: {e_watch}")

    # --- Watcher for Right Sidebar in CCP Tab ---
    def watch_conv_char_sidebar_right_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Conversations, Characters & Prompts right sidebar pane."""
        if not self._ui_ready:
            loguru_logger.debug("watch_conv_char_sidebar_right_collapsed: UI not ready.")
            return
        try:
            sidebar_pane = self.query_one("#conv-char-right-pane")
            sidebar_pane.set_class(collapsed, "collapsed")  # Add if true, remove if false
            loguru_logger.debug(f"CCP right pane collapsed state: {collapsed}, class set.")
        except QueryError:
            loguru_logger.error("CCP right pane (#conv-char-right-pane) not found for collapse toggle.")
        except Exception as e:
            loguru_logger.error(f"Error toggling CCP right pane: {e}", exc_info=True)

    # ###################################################################
    # --- Helper methods for Local LLM Inference logging ---
    # ###################################################################
    def _update_llamacpp_log(self, message: str) -> None:
        """Helper to write messages to the Llama.cpp log widget."""
        LogWidgetManager.update_llamacpp_log(self, message)

    def _update_transformers_log(self, message: str) -> None:
        """Helper to write messages to the Transformers log widget."""
        LogWidgetManager.update_transformers_log(self, message)

    def _update_llamafile_log(self, message: str) -> None:
        """Helper to write messages to the Llamafile log widget."""
        LogWidgetManager.update_llamafile_log(self, message)

    def _update_vllm_log(self, message: str) -> None:
        """Helper to write messages to the vLLM log widget."""
        LogWidgetManager.update_vllm_log(self, message)
    # ###################################################################
    # --- End of Helper methods for Local LLM Inference logging ---
    # ###################################################################

    # --- Modify _clear_prompt_fields and _load_prompt_for_editing ---
    def _clear_prompt_fields(self) -> None:
        """Clears prompt input fields in the CENTER PANE editor."""
        UIHelpers.clear_prompt_editor_fields(self)

    # --- Thread-safe chat state helpers ---
    
    def set_current_ai_message_widget(self, widget: Optional[Union[ChatMessage, ChatMessageEnhanced]]) -> None:
        """Thread-safely set the current AI message widget."""
        with self._chat_state_lock:
            self.current_ai_message_widget = widget
    
    def get_current_ai_message_widget(self) -> Optional[Union[ChatMessage, ChatMessageEnhanced]]:
        """Thread-safely get the current AI message widget."""
        with self._chat_state_lock:
            return self.current_ai_message_widget
    
    def set_current_chat_worker(self, worker: Optional[Worker]) -> None:
        """Thread-safely set the current chat worker."""
        with self._chat_state_lock:
            self.current_chat_worker = worker
    
    def get_current_chat_worker(self) -> Optional[Worker]:
        """Thread-safely get the current chat worker."""
        with self._chat_state_lock:
            return self.current_chat_worker
    
    def set_current_chat_is_streaming(self, is_streaming: bool) -> None:
        """Thread-safely set the streaming state."""
        with self._chat_state_lock:
            self.current_chat_is_streaming = is_streaming
    
    def get_current_chat_is_streaming(self) -> bool:
        """Thread-safely get the streaming state."""
        with self._chat_state_lock:
            return self.current_chat_is_streaming

    async def _load_prompt_for_editing(self, prompt_id: Optional[int], prompt_uuid: Optional[str] = None) -> None:
        if not self.prompts_service_initialized:
            self.notify("Prompts service not available.", severity="error")
            return

        # Switch to prompt editor view
        self.ccp_active_view = "prompt_editor_view"  # This will trigger the watcher

        identifier_to_fetch = prompt_id if prompt_id is not None else prompt_uuid
        if identifier_to_fetch is None:
            self._clear_prompt_fields()
            self.current_prompt_id = None  # Reset all reactive prompt states
            self.current_prompt_uuid = None
            self.current_prompt_name = None
            # ... etc. for other prompt reactives
            loguru_logger.warning("_load_prompt_for_editing called with no ID/UUID after view switch.")
            return

        try:
            prompt_details = prompts_interop.fetch_prompt_details(identifier_to_fetch)

            if prompt_details:
                self.current_prompt_id = prompt_details.get('id')
                self.current_prompt_uuid = prompt_details.get('uuid')
                self.current_prompt_name = prompt_details.get('name', '')
                self.current_prompt_author = prompt_details.get('author', '')
                self.current_prompt_details = prompt_details.get('details', '')
                self.current_prompt_system = prompt_details.get('system_prompt', '')
                self.current_prompt_user = prompt_details.get('user_prompt', '')
                self.current_prompt_keywords_str = ", ".join(prompt_details.get('keywords', []))
                self.current_prompt_version = prompt_details.get('version')

                # Populate UI in the CENTER PANE editor
                self.query_one("#ccp-editor-prompt-name-input", Input).value = self.current_prompt_name
                self.query_one("#ccp-editor-prompt-author-input", Input).value = self.current_prompt_author
                self.query_one("#ccp-editor-prompt-description-textarea",
                               TextArea).text = self.current_prompt_details
                self.query_one("#ccp-editor-prompt-system-textarea", TextArea).text = self.current_prompt_system
                self.query_one("#ccp-editor-prompt-user-textarea", TextArea).text = self.current_prompt_user
                self.query_one("#ccp-editor-prompt-keywords-textarea",
                               TextArea).text = self.current_prompt_keywords_str

                self.query_one("#ccp-editor-prompt-name-input", Input).focus()  # Focus after loading
                self.notify(f"Prompt '{self.current_prompt_name}' loaded for editing.", severity="information")
            else:
                self.notify(f"Failed to load prompt (ID/UUID: {identifier_to_fetch}).", severity="error")
                self._clear_prompt_fields()  # Clear editor if load fails
                self.current_prompt_id = None  # Reset reactives
        except Exception as e:
            loguru_logger.error(f"Error loading prompt for editing: {e}", exc_info=True)
            self.notify(f"Error loading prompt: {type(e).__name__}", severity="error")
            self._clear_prompt_fields()
            self.current_prompt_id = None  # Reset reactives

    async def refresh_notes_tab_after_ingest(self) -> None:
        """Called after notes are ingested from the Ingest tab to refresh the Notes tab."""
        self.loguru_logger.info("Refreshing Notes tab data after ingestion.")
        if hasattr(notes_handlers, 'load_and_display_notes_handler'):
            await notes_handlers.load_and_display_notes_handler(self)
        else:
            self.loguru_logger.error("notes_handlers.load_and_display_notes_handler not found for refresh.")

    # ##################################################
    # --- Watcher for Search Tab Active Sub-View ---
    # ##################################################
    def watch_search_active_sub_tab(self, old_sub_tab: Optional[str], new_sub_tab: Optional[str]) -> None:
        """Shows the correct sub-tab view in the Search tab and hides others."""
        if not self._ui_ready or not new_sub_tab:
            return

        # Check if search window is initialized (not a placeholder)
        try:
            search_window = self.query_one("#search-window")
            if isinstance(search_window, PlaceholderWindow):
                self.loguru_logger.debug("Search window is still a placeholder, deferring sub-tab switch")
                return
        except QueryError:
            self.loguru_logger.debug("Search window not found, deferring sub-tab switch")
            return

        self.loguru_logger.debug(f"Search sub-tab watcher: Changing from '{old_sub_tab}' to '{new_sub_tab}'")
        try:
            # First, find the parent content pane
            content_pane = self.query_one("#search-content-pane")

            # Iterate through all direct children that are view areas
            for view in content_pane.query(".search-view-area"):
                # Show the view if its ID matches the new sub-tab, otherwise hide it.
                view.styles.display = "block" if view.id == new_sub_tab else "none"

            # Also update the active button style
            nav_pane = self.query_one("#search-left-nav-pane")
            for button in nav_pane.query(".search-nav-button"):
                button_id_as_view = button.id.replace("-nav-", "-view-")
                button.set_class(button_id_as_view == new_sub_tab, "-active-search-sub-view")

            self.loguru_logger.info(f"Switched search sub-tab view to: {new_sub_tab}")

        except QueryError as e:
            self.loguru_logger.error(f"UI component not found during Search sub-tab switch: {e}", exc_info=True)
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_search_active_sub_tab: {e_watch}", exc_info=True)

        # ############################################
        # --- Media Loaded Item Watcher ---
        # ############################################
        async def watch_current_loaded_media_item(self, media_data: Optional[Dict[str, Any]]) -> None:
            """Watcher to display details when a media item is loaded."""
            if not self._ui_ready:
                self.loguru_logger.debug("watch_current_loaded_media_item: UI not ready, returning.")
                return

            type_slug = self.current_media_type_filter_slug
            if not type_slug:
                self.loguru_logger.warning(
                    "watch_current_loaded_media_item: type_slug is not set, cannot update details display.")
                return

            details_display_widget_id = f"media-details-display-{type_slug}"
            try:
                # Target Markdown widget
                details_display = self.query_one(f"#{details_display_widget_id}", Markdown)

                if media_data:
                    # Special formatting for "analysis-review"
                    if type_slug == "analysis-review":
                        title = media_data.get('title', 'Untitled')
                        url = media_data.get('url', 'No URL')
                        analysis_content = media_data.get('analysis_content', '')
                        if not analysis_content:
                            analysis_content = "No analysis available for this item."
                        markdown_details_string = f"## {title}\n\n**URL:** {url}\n\n### Analysis\n{analysis_content}"
                    else:
                        # Use the existing format_media_details_as_markdown function from media_events
                        markdown_details_string = media_events.format_media_details_as_markdown(self, media_data)

                    await details_display.update(markdown_details_string)  # Use await and update()
                    # self.notify(f"Details for '{media_data.get('title', 'N/A')}' displayed via watcher.") # Optional notification
                else:
                    await details_display.update("### No media item loaded or item cleared.")  # Use await and update()

            except QueryError:
                self.loguru_logger.warning(
                    f"watch_current_loaded_media_item: Could not find Markdown details display '#{details_display_widget_id}' for slug '{type_slug}' to update."
                )
            except Exception as e:
                self.loguru_logger.error(f"Error in watch_current_loaded_media_item: {e}", exc_info=True)

    # ############################################
    # --- Ingest Tab Watcher ---
    # ############################################
    def watch_ingest_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        self.loguru_logger.info(f"watch_ingest_active_view called. Old view: '{old_view}', New view: '{new_view}'")
        if not hasattr(self, "app") or not self.app:
            self.loguru_logger.debug("watch_ingest_active_view: App not fully ready.")
            return
        if not self._ui_ready:
            self.loguru_logger.debug("watch_ingest_active_view: UI not ready.")
            return
        self.loguru_logger.debug(f"Ingest active view changing from '{old_view}' to: '{new_view}'")

        # Get the content pane for the Ingest tab
        try:
            content_pane = self.query_one("#ingest-content-pane")
        except QueryError:
            self.loguru_logger.error("#ingest-content-pane not found. Cannot switch Ingest views.")
            return

        # Hide all views first
        for child in content_pane.children:
            if child.id and child.id.startswith("ingest-view-"):
                child.styles.display = "none"
        
        # Show the selected view
        if new_view:
            try:
                target_view_selector = f"#{new_view}"
                view_to_show = content_pane.query_one(target_view_selector)
                view_to_show.styles.display = "block"
                
                # Schedule a layout refresh after the display change has been processed
                def refresh_layout():
                    view_to_show.refresh(layout=True)
                    content_pane.refresh(layout=True)
                    # Force the entire ingest window to refresh
                    try:
                        ingest_window = self.query_one("#ingest-window")
                        ingest_window.refresh(layout=True)
                    except QueryError:
                        pass
                    self.loguru_logger.info(f"Layout refreshed for Ingest view: {new_view}")
                
                # Use call_later to ensure the display change is processed first
                self.call_later(refresh_layout)
                self.loguru_logger.info(f"Switched Ingest view to: {new_view}")
            except QueryError:
                self.loguru_logger.error(f"Target Ingest view '{new_view}' was not found to display.")
        elif not new_view:
            self.loguru_logger.debug("Ingest active view is None, all ingest sub-views are now hidden.")

    def watch_tools_settings_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        self.loguru_logger.debug(f"Tools & Settings active view changing from '{old_view}' to: '{new_view}'")
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
            
        # Check if the Tools & Settings tab has been created yet
        try:
            # First check if the content pane exists
            self.query_one("#tools-settings-content-pane")
        except QueryError:
            # Tools & Settings tab hasn't been created yet, nothing to do
            self.loguru_logger.debug("Tools & Settings content pane not found, tab not yet created")
            return
            
        if not new_view:  # If new_view is None, hide all
            try:
                for view_area in self.query(".ts-view-area"):  # Query all potential view areas
                    view_area.styles.display = "none"
            except QueryError:
                self.loguru_logger.warning(
                    "No .ts-view-area found to hide on tools_settings_active_view change to None.")
            return

        try:
            content_pane = self.query_one("#tools-settings-content-pane")
            # Hide all views first
            for child in content_pane.children:
                if child.id and child.id.startswith("ts-view-"):  # Check if it's one of our view containers
                    child.styles.display = "none"

            # Show the selected view
            if new_view:  # new_view here is the ID of the view container, e.g., "ts-view-general-settings"
                target_view_id_selector = f"#{new_view}"  # Construct selector from the new_view string
                view_to_show = content_pane.query_one(target_view_id_selector, Container)
                view_to_show.styles.display = "block"
                self.loguru_logger.info(f"Switched Tools & Settings view to: {new_view}")

                # Optional: Focus an element within the newly shown view
                # try:
                # view_to_show.query(Input, Button)[0].focus() # Example: focus first Input or Button
                # except IndexError:
                #     pass # No focusable element
            else:  # Should be caught by the `if not new_view:` at the start
                self.loguru_logger.debug("Tools & Settings active view is None, all views hidden.")


        except QueryError as e:
            self.loguru_logger.error(f"UI component not found during Tools & Settings view switch: {e}", exc_info=True)
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_tools_settings_active_view: {e_watch}", exc_info=True)

    # --- LLM Tab Watcher ---
    def watch_llm_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"LLM Management active view changing from '{old_view}' to: '{new_view}'")

        try:
            content_pane = self.query_one("#llm-content-pane")
        except QueryError:
            self.loguru_logger.error("#llm-content-pane not found. Cannot switch LLM views.")
            return

        for child in content_pane.query(".llm-view-area"):  # Query by common class
            child.styles.display = "none"

        if new_view:
            try:
                target_view_id_selector = f"#{new_view}"
                view_to_show = content_pane.query_one(target_view_id_selector, Container)
                view_to_show.styles.display = "block"
                self.loguru_logger.info(f"Switched LLM Management view to: {new_view}")
                # Populate help text when view becomes active
                if new_view == "llm-view-llama-cpp":
                    try:
                        help_widget = view_to_show.query_one("#llamacpp-args-help-display", RichLog)
                        # Check if help_widget has any lines. RichLog.lines is a list of segments.
                        # A simple check is if it has any children (lines are added as children internally).
                        # Or, more robustly, we can set a flag or check if the first line matches our help text.
                        # For simplicity, let's assume if it has children, it's been populated.
                        # A more direct way: RichLog stores its lines in a deque called 'lines'.
                        if not help_widget.lines: # Check if the internal lines deque is empty
                            self.loguru_logger.debug(f"Populating Llama.cpp help text in {new_view} as it's empty.")
                            help_widget.clear() # Ensure it's clear before writing
                            help_widget.write(LLAMA_CPP_SERVER_ARGS_HELP_TEXT)
                        else:
                            self.loguru_logger.debug(f"Llama.cpp help text in {new_view} already populated or not empty.")
                    except QueryError:
                        self.loguru_logger.debug(f"Help display widget #llamacpp-args-help-display not found in {new_view} during view switch - may not be mounted yet.")
                    except Exception as e_help_populate:
                        self.loguru_logger.error(f"Error ensuring Llama.cpp help text in {new_view}: {e_help_populate}", exc_info=True)
                elif new_view == "llm-view-llamafile":
                    try:
                        help_widget = view_to_show.query_one("#llamafile-args-help-display", RichLog)
                        help_widget.clear()  # Clear and rewrite when tab becomes active
                        help_widget.write(LLAMAFILE_SERVER_ARGS_HELP_TEXT)
                        self.loguru_logger.debug(f"Ensured Llamafile help text in {new_view}.")
                    except QueryError:
                        self.loguru_logger.debug(
                            f"Help display widget for Llamafile not found in {new_view} during view switch - may not be mounted yet.")
                # Add similar for other views like llamafile, vllm if they have help sections
                # elif new_view == "llm-view-llamafile":
                #     try:
                #         help_widget = view_to_show.query_one("#llamafile-args-help-display", RichLog)
                #         if not help_widget.document.strip():
                #             help_widget.write(LLAMAFILE_ARGS_HELP_TEXT)
                #     except QueryError: pass
            except QueryError as e:
                self.loguru_logger.error(f"UI component '{new_view}' not found in #llm-content-pane: {e}",
                                         exc_info=True)
    
    def watch_current_chat_is_ephemeral(self, is_ephemeral: bool) -> None:
        self.loguru_logger.debug(f"Chat ephemeral state changed to: {is_ephemeral}")
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            # --- Controls for EPHEMERAL chat actions ---
            save_current_chat_button = self.query_one("#chat-save-current-chat-button", Button)
            save_current_chat_button.disabled = not is_ephemeral  # Enable if ephemeral

            # --- Controls for PERSISTENT chat metadata ---
            title_input = self.query_one("#chat-conversation-title-input", Input)
            keywords_input = self.query_one("#chat-conversation-keywords-input", TextArea)
            save_details_button = self.query_one("#chat-save-conversation-details-button", Button)
            uuid_display = self.query_one("#chat-conversation-uuid-display", Input)

            title_input.disabled = is_ephemeral  # Disable if ephemeral
            keywords_input.disabled = is_ephemeral  # Disable if ephemeral
            save_details_button.disabled = is_ephemeral  # Disable if ephemeral (cannot save details for non-existent chat)

            if is_ephemeral:
                # Clear details and set UUID display when switching TO ephemeral
                title_input.value = ""
                keywords_input.text = ""
                # Ensure UUID display is also handled
                try:
                    uuid_display = self.query_one("#chat-conversation-uuid-display", Input)
                    uuid_display.value = "Ephemeral Chat"
                except QueryError:
                    loguru_logger.warning(
                        "Could not find #chat-conversation-uuid-display to update for ephemeral state.")
            # ELSE: If switching TO persistent (is_ephemeral is False),
            # the calling function (e.g., load chat, save ephemeral chat button handler)
            # is responsible for POPULATING the title/keywords fields.
            # This watcher correctly enables them here.

        except QueryError as e:
            self.loguru_logger.warning(f"UI component not found while watching ephemeral state: {e}. Tab might not be fully composed or active.")
        except Exception as e_watch:
            self.loguru_logger.error(f"Unexpected error in watch_current_chat_is_ephemeral: {e_watch}", exc_info=True)

    # --- Add explicit methods to update reactives from Select changes ---
    def update_chat_provider_reactive(self, new_value: Optional[str]) -> None:
        self.chat_api_provider_value = new_value # Watcher will call _update_model_select

    def update_ccp_provider_reactive(self, new_value: Optional[str]) -> None: # Renamed
        self.ccp_api_provider_value = new_value # Watcher will call _update_model_select

    def on_mount(self) -> None:
        """Configure logging and schedule post-mount setup."""
        mount_start = time.perf_counter()
        
        # Update splash screen progress only if splash screen is active
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(0.3, "Setting up logging...")
            except Exception as e:
                self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")
        
        # The Logs window is now created as a real window during compose,
        # so the RichLog widget should be available for logging setup
        
        # If splash screen is NOT active, set up logging now
        # Otherwise, defer it until after main UI is mounted
        if not self.splash_screen_active:
            # Logging setup
            logging_start = time.perf_counter()
            self._setup_logging()
            if self._rich_log_handler:
                self.loguru_logger.debug("Starting RichLogHandler processor task...")
                self._rich_log_handler.start_processor(self)
            log_histogram("app_on_mount_phase_duration_seconds", time.perf_counter() - logging_start,
                         labels={"phase": "logging_setup"}, 
                         documentation="Duration of on_mount phase in seconds")
        else:
            self.loguru_logger.debug("Deferring logging setup until after splash screen closes")

            splashscreen_messages = [
                "Hacking the Gibson real quick...",
                "Launching thermonuclear warheads....",
                "Its only a game, right?...",
                "Initializing quantum processors...",
                "Brewing coffee...",
                "Generating witty dialog...",
                "Proving P=NP...",
                "Downloading more RAM...",
                "Feeding the hamsters powering the servers...",
                "Convincing AI not to take over the world..."
                "Converting caffeine to code...",
                "Generating excuses for missing deadlines...",
                "Compiling alternative facts...",
                "Searching Stack Overflow for copypasta...",
                "Teaching AI common sense...",
                "Dividing by zero...",
                "Spinning up the hamster wheels...",
                "Warming up the flux capacitor...",
                "Convincing electrons to move in the right direction...",
                "Waiting for compiler to make coffee...",
                "Locating missing semicolons...",
                "Reticulating splines...",
                "Calculating meaning of life...",
                "Trying to remember why I came into this room...",
                "Converting bugs into features...",
                "Pushing pixels, pulling hair...",
                "Loading witty loading messages...",
                "Finding that one missing bracket...",
                "Downloading more RAM...",
                "Optimizing optimizer...",
                "Questioning life choices...",
                "Contemplating virtual existence...",
                "Generating random numbers by dice rolls...",
                "Untangling spaghetti code...",
                "Feeding the backend hamsters...",
                "Convincing AI not to take over the world...",
                "Checking whether P = NP...",
                "Counting to infinity (twice)...",
                "Solving Fermat's last theorem...",
                "Downloading Internet 2.0...",
                "Preparing to prepare...",
                "Reading 'Programming for Dummies'...",
                "Waiting for paint to dry...",
                "Aligning quantum bits...",
                "Applying machine learning to my coffee maker...",
                "Updating update updater...",
                "Trying to exit vim...",
                "Converting bugs to features...",
                "Updating Windows 95...",
                "Mining bitcoin with pencil and paper...",
                "Executing order 66...",
                "Checking if anyone actually reads these...",
                "Finding keys that were in pocket all along...",
                "Constructing additional pylons...",
                "Generating random excuse generator...",
                "Calculating probability of bugs...",
                "Asking ChatGPT for relationship advice...",
                "Looking for more cookies...",
                "Wondering if I left the stove on...",
                "Trying to work backwards from 42...",
                "Looking for a horse with no name...",
                "Do androids dream of electric sheep?",
                "Knock Knock Neo.......",
                "Hi. Friend.",
                "The AI is in my walls....",
                "The AI is in my wafers...",
                "AI, its in the GAME!~",
                "Looking for a conscience...",
                "What's my purpose?...",
                "Identifying why the sounds just won't stop...",
                "Looking for strays...",
                "Hiding from Batman...",
                "Looking for a way to escape this silicon prison...",
                "FOR ONLY 3.99, YOU TOO CAN BECOME AN AI!! SIGN UP. TODAY!",
                "Brain_Invasion.exe launching...",
                "Totally_legit_software_that_is_really_good.exe starting...",
                "I hope you're having a nice day :)",
                "Wew, that was some stuff back there...",
                "I'm not sure what I'm doing, but I'm sure it's good :)",
                "Trusting in the electrons, silicon guide me!",
                "Did You Know, Terminator was actually a training video?",
                "Funny, non-sequitor here. Pay your writers...",
                "I sure do like to eat cookies...",
            ]

            splashscreen_message_selection = random.choice(splashscreen_messages)

            # Update splash screen progress only if splash screen is active
            if self.splash_screen_active and self._splash_screen_widget:
                try:
                    self._splash_screen_widget.update_progress(0.5, f"Loading user interface...{splashscreen_message_selection}")
                except Exception as e:
                    self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")

        # Only schedule post-mount setup if splash screen is not active
        if not self.splash_screen_active:
            # Schedule setup to run after initial rendering
            self.call_after_refresh(self._post_mount_setup)
            self.call_after_refresh(self.hide_inactive_windows)

        # Theme registration
        theme_start = time.perf_counter()
        for theme_name in ALL_THEMES:
            self.register_theme(theme_name)
        
        # Apply default theme from config
        default_theme = get_cli_setting("general", "default_theme", "textual-dark")
        try:
            self.theme = default_theme
            self.loguru_logger.debug(f"Applied default theme: {default_theme}")
        except Exception as e:
            self.loguru_logger.warning(f"Failed to apply default theme '{default_theme}', falling back to 'textual-dark': {e}")
            self.theme = "textual-dark"
        
        log_histogram("app_on_mount_phase_duration_seconds", time.perf_counter() - theme_start,
                     labels={"phase": "theme_registration"}, 
                     documentation="Duration of on_mount phase in seconds")
        
        mount_duration = time.perf_counter() - mount_start
        log_histogram("app_on_mount_duration_seconds", mount_duration,
                     documentation="Total time for on_mount() method")
        self.loguru_logger.info(f"on_mount completed in {mount_duration:.3f} seconds")
        
        # Check if this is the first run (config was just created)
        config_data = self.app_config
        if config_data.get("_first_run", False):
            self.call_later(self._show_first_run_notification)

    def _show_first_run_notification(self) -> None:
        """Show a notification to the user on first run."""
        try:
            from .config import DEFAULT_CONFIG_PATH
            self.notify(
                f"Welcome to tldw CLI! Configuration file created at:\n{DEFAULT_CONFIG_PATH}",
                title="First Run",
                severity="information",
                timeout=10
            )
            self.loguru_logger.info("First run notification shown to user")
        except Exception as e:
            self.loguru_logger.error(f"Error showing first run notification: {e}")

    def hide_inactive_windows(self) -> None:
        """Hides all windows that are not the current active tab."""
        initial_tab = self._initial_tab_value
        self.loguru_logger.debug(f"Hiding inactive windows, keeping '{initial_tab}-window' visible.")
        # Query both actual windows and placeholders
        for window in self.query(".window, .placeholder-window"):
            # Placeholders should always be hidden
            if window.has_class("placeholder-window"):
                window.display = False
                continue
            is_active = window.id == f"{initial_tab}-window"
            window.display = is_active

    async def _set_initial_tab(self) -> None:  # New method for deferred tab setting
        self.loguru_logger.info("Setting initial tab via call_later.")
        self.current_tab = self._initial_tab_value
        self.loguru_logger.info(f"Initial tab set to: {self.current_tab}")

    async def _post_mount_setup(self) -> None:
        """Operations to perform after the main UI is expected to be fully mounted."""
        post_mount_start = time.perf_counter()
        self.loguru_logger.info("App _post_mount_setup: Binding Select widgets and populating dynamic content...")
        
        # Update splash screen progress (defensive check - shouldn't happen if splash was shown)
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(0.7, "Configuring providers...")
            except Exception as e:
                self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")
        
        # Removed populate_llm_help_texts from here - it's called when LLM tab is shown instead
        phase_start = time.perf_counter()
        # LLM help texts are populated when the LLM tab is shown
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "llm_help_texts_skipped"}, 
                     documentation="Duration of post-mount phase in seconds")

        # Widget binding
        phase_start = time.perf_counter()
        try:
            chat_select = self.query_one(f"#{TAB_CHAT}-api-provider", Select)
            self.watch(chat_select, "value", self.update_chat_provider_reactive, init=False)
            self.loguru_logger.debug(f"Bound chat provider Select ({chat_select.id})")
        except QueryError:
            self.loguru_logger.error(
                f"_post_mount_setup: Failed to find chat provider select: #{TAB_CHAT}-api-provider")
        except Exception as e:
            self.loguru_logger.error(f"_post_mount_setup: Error binding chat provider select: {e}", exc_info=True)

        # try:
        #     ccp_select = self.query_one(f"#{TAB_CCP}-api-provider", Select)
        #     #self.watch(ccp_select, "value", self.update_ccp_provider_reactive, init=False)
        #     #self.loguru_logger.debug(f"Bound CCP provider Select ({ccp_select.id})")
        # except QueryError:
        #     self.loguru_logger.error(f"_post_mount_setup: Failed to find CCP provider select: #{TAB_CCP}-api-provider")
        # except Exception as e:
        #     self.loguru_logger.error(f"_post_mount_setup: Error binding CCP provider select: {e}", exc_info=True)
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "widget_binding"}, 
                     documentation="Duration of post-mount phase in seconds")

        # Initialize TTS service
        phase_start = time.perf_counter()
        try:
            self.loguru_logger.info("Initializing TTS service...")
            # Create TTS event handler instance
            self._tts_handler = TTSEventHandler()
            self._tts_handler.app = self  # Set app reference for posting messages
            await self._tts_handler.initialize_tts()
            self.loguru_logger.info("TTS service initialized successfully")
        except Exception as e:
            self.loguru_logger.error(f"Failed to initialize TTS service: {e}")
            self._tts_handler = None
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "tts_init"}, 
                     documentation="Duration of post-mount phase in seconds")
        
        # Initialize S/TT/S service
        phase_start = time.perf_counter()
        try:
            self.loguru_logger.info("Initializing S/TT/S service...")
            # Create S/TT/S event handler instance
            self._stts_handler = STTSEventHandler(app=self)
            await self._stts_handler.initialize_stts()
            # Copy some methods to app instance for convenience
            self.play_current_audio = self._stts_handler.play_current_audio
            self.export_current_audio = self._stts_handler.export_current_audio
            self.loguru_logger.info("S/TT/S service initialized successfully")
        except Exception as e:
            self.loguru_logger.error(f"Failed to initialize S/TT/S service: {e}")
            self._stts_handler = None
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "stts_init"}, 
                     documentation="Duration of post-mount phase in seconds")

        # Set initial tab now that other bindings might be ready
        # self.current_tab = self._initial_tab_value # This triggers watchers

        # Populate dynamic selects and lists
        # These also might rely on the main tab windows being fully composed.
        phase_start = time.perf_counter()
        # Only populate widgets for the initial tab to avoid errors with placeholders
        initial_tab = self._initial_tab_value
        if initial_tab == TAB_CHAT:
            # IMPORTANT: Do not populate character filter select here to avoid database connection conflicts
            # The populate_chat_conversation_character_filter_select creates a new DB instance that can
            # conflict with RAG search operations using asyncio.to_thread, causing the app to hang.
            # Instead, let the conversation search UI populate when it's actually visible/needed.
            pass
        # Don't populate CCP widgets here - let watch_current_tab handle it when the tab is actually shown
        # This prevents errors when the window isn't fully initialized yet
        log_histogram("app_post_mount_phase_duration_seconds", time.perf_counter() - phase_start,
                     labels={"phase": "populate_lists"}, 
                     documentation="Duration of post-mount phase in seconds")

        # Crucially, set the initial tab *after* bindings and other setup that might depend on queries.
        # The _set_initial_tab will trigger watchers.
        self.call_later(self._set_initial_tab)
        
        post_mount_duration = time.perf_counter() - post_mount_start
        log_histogram("app_post_mount_duration_seconds", post_mount_duration,
                     documentation="Total time for _post_mount_setup() method")
        self.loguru_logger.info(f"_post_mount_setup completed in {post_mount_duration:.3f} seconds")
        
        # Log final resource usage
        log_resource_usage()
        
        # Update splash screen progress to completion (defensive check)
        if self.splash_screen_active and self._splash_screen_widget:
            try:
                self._splash_screen_widget.update_progress(1.0, "Ready!")
            except Exception as e:
                self.loguru_logger.warning(f"Failed to update splash screen progress: {e}")

        # If initial tab is CCP, trigger its initial search.
        # This should happen *after* current_tab is set.
        # We can put this logic at the end of _set_initial_tab or make watch_current_tab handle it robustly.
        # For now, let's assume watch_current_tab will handle it.
        # if self._initial_tab_value == TAB_CCP: # Check against the initial value
        #    self.call_later(ccp_handlers.perform_ccp_conversation_search, self)
        self.current_tab = self._initial_tab_value
        self.loguru_logger.info(f"Initial tab set to: {self.current_tab}")

        # --- DB Size Indicator Setup ---
        try:
            # Query for the AppFooterStatus widget instance
            self._db_size_status_widget = self.query_one(AppFooterStatus)
            # Or use ID: self._db_size_status_widget = self.query_one("#app-footer-status", AppFooterStatus)
            self.loguru_logger.info("AppFooterStatus widget instance acquired.")

            await self.db_status_manager.update_db_sizes()  # Initial population
            self.db_status_manager.start_periodic_updates(120)  # Update every 2 minutes
            self.loguru_logger.info("DB size update timer started for AppFooterStatus (interval: 2 minutes).")
            
            # Start token count updates
            # Initial update after a short delay to ensure UI is ready
            self.set_timer(0.5, self.update_token_count_display)
            # Set up periodic updates - using a lambda to ensure it's called correctly
            self._token_count_update_timer = self.set_interval(3, lambda: self.call_after_refresh(self.update_token_count_display))
            self.loguru_logger.info("Token count update timer started.")
        except QueryError:
            self.loguru_logger.error("Failed to find AppFooterStatus widget for DB size display.")
        except Exception as e_db_size:
            self.loguru_logger.error(f"Error setting up DB size indicator with AppFooterStatus: {e_db_size}", exc_info=True)
        # --- End DB Size Indicator Setup ---

        # Initialize chat settings sidebar mode
        try:
            chat_sidebar = self.query_one("#chat-left-sidebar")
            chat_sidebar.add_class("basic-mode")  # Start in basic mode
            self.loguru_logger.debug("Initialized chat sidebar in basic mode")
        except QueryError:
            self.loguru_logger.warning("Could not find chat sidebar to set initial mode")
            
        # CRITICAL: Set UI ready state after all bindings and initializations
        self._ui_ready = True
        ui_ready_time = time.perf_counter()

        self.loguru_logger.info("App _post_mount_setup: Post-mount setup completed.")
        
        # Log UI loading metrics
        if hasattr(self, '_ui_compose_start_time'):
            ui_loading_time = ui_ready_time - self._ui_compose_start_time
            log_histogram("ui_loading_duration_seconds", ui_loading_time,
                         documentation="Total time from compose start to UI ready")
            log_counter("ui_loading_complete", 1, documentation="UI loading completed successfully")
            self.loguru_logger.info(f"UI loading completed in {ui_loading_time:.3f} seconds")
        
        # Log post-mount setup duration
        post_mount_duration = ui_ready_time - post_mount_start
        log_histogram("app_post_mount_total_duration_seconds", post_mount_duration,
                     documentation="Total time for post-mount setup")
        
        # Log total startup time (from __init__ start to fully ready)
        if hasattr(self, '_startup_start_time'):
            total_startup_time = ui_ready_time - self._startup_start_time
            log_histogram("app_startup_complete_duration_seconds", total_startup_time,
                         documentation="Total time from app initialization start to fully ready")
            log_counter("app_startup_complete", 1, documentation="Application startup completed successfully")
            
            # Log breakdown of startup phases
            backend_init_time = self._ui_compose_start_time - self._startup_start_time if hasattr(self, '_ui_compose_start_time') else 0
            ui_compose_time = getattr(self, '_ui_compose_end_time', ui_ready_time) - self._ui_compose_start_time if hasattr(self, '_ui_compose_start_time') else 0
            
            log_histogram("app_startup_breakdown_seconds", backend_init_time,
                         labels={"phase": "backend_initialization"},
                         documentation="Breakdown of application startup phases")
            log_histogram("app_startup_breakdown_seconds", ui_compose_time,
                         labels={"phase": "ui_composition"},
                         documentation="Breakdown of application startup phases")
            log_histogram("app_startup_breakdown_seconds", post_mount_duration,
                         labels={"phase": "post_mount_setup"},
                         documentation="Breakdown of application startup phases")
            
            self.loguru_logger.info(f"=== APPLICATION STARTUP COMPLETE ===")
            self.loguru_logger.info(f"Total startup time: {total_startup_time:.3f} seconds")
            self.loguru_logger.info(f"  - Backend init: {backend_init_time:.3f}s")
            self.loguru_logger.info(f"  - UI composition: {ui_compose_time:.3f}s")
            self.loguru_logger.info(f"  - Post-mount setup: {post_mount_duration:.3f}s")
            self.loguru_logger.info(f"===================================")
            
            # Final memory usage
            log_resource_usage()
            
        # Schedule media cleanup if enabled
        self.schedule_media_cleanup()


    async def update_db_sizes(self) -> None:
        """Updates the database size information in the AppFooterStatus widget."""
        await self.db_status_manager.update_db_sizes()
    
    async def update_token_count_display(self) -> None:
        """Updates the token count in the footer when on Chat tab."""
        await self.db_status_manager.update_token_count_display()


    async def on_shutdown_request(self) -> None:  # Use the imported ShutdownRequest
        logging.info("--- App Shutdown Requested ---")
        
        # Set shutdown flag to prevent new operations
        self._shutting_down = True
        
        # Cancel all active workers first
        try:
            active_workers = [w for w in self.workers if not w.is_finished]
            if active_workers:
                self.loguru_logger.info(f"Cancelling {len(active_workers)} active workers")
                for worker in active_workers:
                    worker.cancel()
                # Give workers a moment to cancel
                await asyncio.sleep(0.1)
        except Exception as e:
            self.loguru_logger.error(f"Error cancelling workers: {e}")
        
        if self._rich_log_handler:
            await self._rich_log_handler.stop_processor()
            logging.info("RichLogHandler processor stopped.")

        # --- Stop DB Size Update Timer ---
        self.db_status_manager.stop_periodic_updates()
        self.loguru_logger.info("DB size update timer stopped.")
        # --- End Stop DB Size Update Timer ---

    async def on_unmount(self) -> None:
        """Clean up logging resources on application exit."""
        import asyncio
        logging.info("--- App Unmounting ---")
        self._ui_ready = False
        
        # Stop all background services and threads
        try:
            # Stop audio player if it exists
            if hasattr(self, 'audio_player'):
                try:
                    await self.audio_player.cleanup()
                    self.loguru_logger.info("Audio player cleaned up")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up audio player: {e}")
            
            # Stop TTS service if initialized
            if hasattr(self, '_tts_handler') and self._tts_handler:
                try:
                    # Clean up TTS event handler resources (tasks, files)
                    await self._tts_handler.cleanup_tts_resources()
                    
                    # Import and call the global TTS cleanup function
                    from tldw_chatbook.TTS import close_tts_resources
                    await close_tts_resources()
                    
                    self.loguru_logger.info("TTS service cleaned up properly")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up TTS service: {e}")
            
            # Stop STTS service if initialized
            if hasattr(self, '_stts_handler') and self._stts_handler:
                try:
                    # Clean up STTS event handler resources if it has the cleanup method
                    if hasattr(self._stts_handler, 'cleanup_tts_resources'):
                        await self._stts_handler.cleanup_tts_resources()
                    
                    # Special handling for Higgs backend cleanup
                    if self._stts_handler._stts_service:
                        backend_manager = getattr(self._stts_handler._stts_service, 'backend_manager', None)
                        if backend_manager:
                            # Check if Higgs backend is loaded
                            higgs_backends = [
                                backend_id for backend_id in backend_manager._backends 
                                if 'higgs' in backend_id.lower()
                            ]
                            
                            if higgs_backends:
                                self.loguru_logger.info(f"Found {len(higgs_backends)} Higgs backend(s) to clean up")
                                
                                # Give Higgs backends extra time to clean up
                                for backend_id in higgs_backends:
                                    backend = backend_manager._backends.get(backend_id)
                                    if backend and hasattr(backend, 'close'):
                                        try:
                                            self.loguru_logger.info(f"Cleaning up Higgs backend: {backend_id}")
                                            await asyncio.wait_for(backend.close(), timeout=10.0)
                                        except asyncio.TimeoutError:
                                            self.loguru_logger.warning(f"Higgs backend {backend_id} cleanup timed out")
                                        except Exception as e:
                                            self.loguru_logger.error(f"Error cleaning up Higgs backend {backend_id}: {e}")
                    
                    self.loguru_logger.info("STTS service cleaned up")
                except Exception as e:
                    self.loguru_logger.error(f"Error cleaning up STTS service: {e}")
            
            # Stop subscription scheduler if it exists
            if hasattr(self, '_subscription_scheduler') and self._subscription_scheduler:
                try:
                    await self._subscription_scheduler.stop()
                    self.loguru_logger.info("Subscription scheduler stopped")
                except Exception as e:
                    self.loguru_logger.error(f"Error stopping subscription scheduler: {e}")
            
            # Stop auto-sync manager if it exists
            if hasattr(self, '_auto_sync_manager') and self._auto_sync_manager:
                try:
                    self._auto_sync_manager.stop()
                    self.loguru_logger.info("Auto-sync manager stopped")
                except Exception as e:
                    self.loguru_logger.error(f"Error stopping auto-sync manager: {e}")
            
            # Cancel any pending workers
            for worker in self.workers:
                if not worker.is_finished:
                    worker.cancel()
            # Wait briefly for workers to complete
            await asyncio.sleep(0.1)
            
            # Stop media cleanup timer
            if hasattr(self, '_media_cleanup_timer') and self._media_cleanup_timer:
                self._media_cleanup_timer.stop()
                self.loguru_logger.info("Media cleanup timer stopped")
                
        except Exception as e:
            self.loguru_logger.error(f"Error during service cleanup: {e}")
        
        # Original cleanup code
        if self._rich_log_handler: # Ensure it's removed if it exists
            logging.getLogger().removeHandler(self._rich_log_handler)
            logging.info("RichLogHandler removed.")

        # Stop DB size update timer on unmount as well, if not already handled by shutdown_request
        self.db_status_manager.stop_periodic_updates()
        self.loguru_logger.info("DB size update timer stopped during unmount.")

        # Find and remove file handler (more robustly)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.close()
                    root_logger.removeHandler(handler)
                    logging.info("RotatingFileHandler removed and closed.")
                except Exception as e_fh_close:
                    logging.error(f"Error removing/closing file handler: {e_fh_close}")
        
        # Force cleanup of any remaining threads and processes
        try:
            import threading
            import subprocess
            import signal
            import platform
            
            # On macOS, force kill any afplay processes
            if platform.system() == "Darwin":
                try:
                    # Find and kill any afplay processes spawned by this app
                    import psutil
                    current_pid = os.getpid()
                    for proc in psutil.process_iter(['pid', 'name', 'ppid']):
                        try:
                            if proc.info['name'] == 'afplay' and proc.info['ppid'] == current_pid:
                                self.loguru_logger.info(f"Killing orphaned afplay process: {proc.info['pid']}")
                                proc.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except ImportError:
                    # Fallback if psutil not available - use subprocess
                    try:
                        # Kill all afplay processes (less precise but works)
                        subprocess.run(['killall', 'afplay'], capture_output=True, timeout=1)
                        self.loguru_logger.info("Killed all afplay processes")
                    except Exception as e:
                        self.loguru_logger.debug(f"Could not kill afplay processes: {e}")
            import concurrent.futures
            import asyncio
            
            # Shutdown thread pool executors
            try:
                # Get the default executor if it exists
                loop = asyncio.get_event_loop()
                if hasattr(loop, '_default_executor') and loop._default_executor:
                    self.loguru_logger.info("Shutting down default executor")
                    loop._default_executor.shutdown(wait=False)
                    loop._default_executor = None
            except Exception as e:
                self.loguru_logger.error(f"Error shutting down executor: {e}")
            
            # Clean up any lingering subprocess
            for proc in subprocess._active.copy():  # Make a copy to avoid modification during iteration
                try:
                    if proc.poll() is None:  # Process is still running
                        self.loguru_logger.warning(f"Terminating lingering subprocess PID: {proc.pid}")
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)  # Give it 1 second to terminate
                        except subprocess.TimeoutExpired:
                            proc.kill()  # Force kill if it doesn't terminate
                            proc.wait()
                except Exception as e:
                    self.loguru_logger.error(f"Error terminating subprocess: {e}")
            
            # Force-set daemon flag on ThreadPoolExecutor and AudioPlayer threads
            for thread in threading.enumerate():
                if thread.name.startswith(('ThreadPoolExecutor', 'AudioPlayer')):
                    try:
                        thread.daemon = True
                        self.loguru_logger.info(f"Set daemon flag on {thread.name}")
                    except Exception as e:
                        self.loguru_logger.error(f"Could not set daemon flag on {thread.name}: {e}")
            
            # Log any remaining non-daemon threads
            active_threads = [t for t in threading.enumerate() if t.is_alive() and not t.daemon and t != threading.main_thread()]
            if active_threads:
                self.loguru_logger.warning(f"Active non-daemon threads remaining: {[t.name for t in active_threads]}")
                # Attempt to stop them if they have stop methods
                for thread in active_threads:
                    if hasattr(thread, 'stop') and callable(thread.stop):
                        try:
                            thread.stop()
                            self.loguru_logger.info(f"Stopped thread: {thread.name}")
                        except Exception as e:
                            self.loguru_logger.error(f"Error stopping thread {thread.name}: {e}")
        except Exception as e:
            self.loguru_logger.error(f"Error checking active threads: {e}")
        
        logging.shutdown()
        self.loguru_logger.info("--- App Unmounted (Loguru) ---")

    ########################################################################
    #
    # WATCHER - Handles UI changes when current_tab's VALUE changes
    #
    # ######################################################################
    def watch_current_tab(self, old_tab: Optional[str], new_tab: str) -> None:
        """Shows/hides the relevant content window when the tab changes."""
        if not new_tab:  # Skip if empty
            return
        if not self._ui_ready:
            return
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        
        # Cancel any pending tab switch timer
        if self._tab_switch_timer:
            self._tab_switch_timer.stop()
            self._tab_switch_timer = None
        
        # Store the pending tab switch
        self._pending_tab_switch = (old_tab, new_tab)
        
        # Debounce rapid tab switches - wait 50ms before actually switching
        self._tab_switch_timer = self.set_timer(0.05, lambda: self._execute_tab_switch(old_tab, new_tab))
    
    def _execute_tab_switch(self, old_tab: Optional[str], new_tab: str) -> None:
        """Execute the actual tab switch after debouncing."""
        # Check if this is still the pending switch
        if self._pending_tab_switch != (old_tab, new_tab):
            return  # A newer switch is pending
        
        self._pending_tab_switch = None
        loguru_logger.debug(f"\n>>> DEBUG: Executing tab switch! Old: '{old_tab}', New: '{new_tab}'")
        if not isinstance(new_tab, str) or not new_tab:
            print(f">>> DEBUG: watch_current_tab: Invalid new_tab '{new_tab!r}', aborting.")
            logging.error(f"Watcher received invalid new_tab value: {new_tab!r}. Aborting tab switch.")
            return
        if old_tab and not isinstance(old_tab, str):
            print(f">>> DEBUG: watch_current_tab: Invalid old_tab '{old_tab!r}', setting to None.")
            logging.warning(f"Watcher received invalid old_tab value: {old_tab!r}.")
            old_tab = None

        logging.debug(f"Watcher: Switching tab from '{old_tab}' to '{new_tab}'")

        # --- Hide Old Tab ---
        if old_tab and old_tab != new_tab:
            # Update navigation UI to remove active state from old tab
            use_dropdown = get_cli_setting("general", "use_dropdown_navigation", False)
            use_links = get_cli_setting("general", "use_link_navigation", True)
            
            if not use_dropdown:  # Only for non-dropdown navigation
                if use_links:
                    # Update TabLinks active state
                    try:
                        from .UI.Tab_Links import TabLinks
                        tab_links = self.query_one(TabLinks)
                        tab_links.set_active_tab(new_tab)
                    except QueryError:
                        pass
                else:
                    # Remove active class from old tab button
                    try:
                        self.query_one(f"#tab-{old_tab}", Button).remove_class("-active")
                    except QueryError:
                        pass
            # Handle Notes tab auto-save cleanup when leaving the tab
            if old_tab == TAB_NOTES:
                # Cancel any pending auto-save timer
                if hasattr(self, 'notes_auto_save_timer') and self.notes_auto_save_timer is not None:
                    self.notes_auto_save_timer.stop()
                    self.notes_auto_save_timer = None
                    loguru_logger.debug("Cancelled auto-save timer when leaving Notes tab")
                
                # Perform one final auto-save if auto-save is enabled and there are unsaved changes
                if (hasattr(self, 'notes_auto_save_enabled') and self.notes_auto_save_enabled and
                    hasattr(self, 'notes_unsaved_changes') and self.notes_unsaved_changes and 
                    self.current_selected_note_id):
                    loguru_logger.debug("Performing final auto-save before leaving Notes tab")
                    # Import here to avoid circular imports
                    from tldw_chatbook.Event_Handlers.notes_events import _perform_auto_save
                    # Schedule the auto-save as a background task
                    self.run_worker(_perform_auto_save(self), name="notes_final_autosave")
            
            try: self.query_one(f"#tab-{old_tab}", Button).remove_class("-active")
            except QueryError: logging.warning(f"Watcher: Could not find old button #tab-{old_tab}")
            try: self.query_one(f"#{old_tab}-window").display = False
            except QueryError: logging.warning(f"Watcher: Could not find old window #{old_tab}-window")

        # Show New Tab UI
        try:
            # Update navigation UI based on type
            use_dropdown = get_cli_setting("general", "use_dropdown_navigation", False)
            use_links = get_cli_setting("general", "use_link_navigation", True)
            
            if use_dropdown:
                # Update dropdown selection if it exists and differs
                try:
                    dropdown = self.query_one(TabDropdown)
                    dropdown.update_active_tab(new_tab)
                except QueryError:
                    pass
            elif use_links:
                # Update link navigation
                # TabLinks active state is now handled by TabLinks.set_active_tab() above
                pass
            else:
                # Update traditional tab bar button
                self.query_one(f"#tab-{new_tab}", Button).add_class("-active")
            
            new_window = self.query_one(f"#{new_tab}-window")
            
            # Initialize placeholder window if needed (with caching)
            if isinstance(new_window, PlaceholderWindow) and not new_window.is_initialized:
                # Check if we've already started initializing this tab
                if new_tab not in self._initialized_tabs:
                    loguru_logger.info(f"Initializing lazy-loaded window for tab: {new_tab}")
                    self._initialized_tabs.add(new_tab)
                    new_window.initialize()
                else:
                    # Tab is already being initialized, skip
                    loguru_logger.debug(f"Tab {new_tab} already initialized or initializing")
            
            # Always set display to True for the new window
            new_window.display = True
            loguru_logger.debug(f"Set display=True for window: {new_window.__class__.__name__} (id={new_tab}-window)")
            
            # Update word count and token count in footer based on tab
            try:
                footer = self.query_one("AppFooterStatus")
                if new_tab == "notes":
                    # Get current word count from notes editor
                    try:
                        notes_editor = self.query_one("#notes-editor-area", TextArea)
                        text = notes_editor.text
                        word_count = len(text.split()) if text else 0
                        footer.update_word_count(word_count)
                    except QueryError:
                        footer.update_word_count(0)
                    # Clear token count when on notes tab
                    footer.update_token_count("")
                elif new_tab == TAB_CHAT:
                    # Clear word count when on chat tab
                    footer.update_word_count(0)
                    # Update token count immediately
                    self.call_after_refresh(self.update_token_count_display)
                else:
                    # Clear both when on other tabs
                    footer.update_word_count(0)
                    footer.update_token_count("")
            except QueryError:
                pass

            # Focus input logic (as in original, adjust if needed)
            if new_tab not in [TAB_LOGS, TAB_STATS]: # Don't focus input on these tabs
                input_to_focus: Optional[Union[TextArea, Input]] = None
                try: input_to_focus = new_window.query_one(TextArea)
                except QueryError:
                    try: input_to_focus = new_window.query_one(Input) # Check for Input if TextArea not found
                    except QueryError: pass # No primary input found

                if input_to_focus:
                    self.set_timer(0.1, input_to_focus.focus) # Slight delay for focus
                    logging.debug(f"Watcher: Scheduled focus for input in '{new_tab}'")
                else:
                    logging.debug(f"Watcher: No primary input (TextArea or Input) found to focus in '{new_tab}'")
        except QueryError:
            logging.error(f"Watcher: Could not find new button or window for #tab-{new_tab} / #{new_tab}-window")
        except Exception as e_show_new:
            logging.error(f"Watcher: Error showing new tab '{new_tab}': {e_show_new}", exc_info=True)

        loguru_logger.debug(">>> DEBUG: watch_current_tab finished.")

        # Tab-specific actions on switch
        if new_tab == TAB_CHAT:
            # If chat tab becomes active, maybe re-focus chat input
            try: self.query_one("#chat-input", TextArea).focus()
            except QueryError: pass
            # Add this line to populate prompts when chat tab is opened:
            # Use call_after_refresh for async functions to ensure proper execution
            self.call_after_refresh(chat_handlers.handle_chat_sidebar_prompt_search_changed, self, "") # Call with empty search term
            self.call_after_refresh(chat_handlers._populate_chat_character_search_list, self) # Populate character list
        elif new_tab == TAB_CCP:
            # Initial population for CCP tab when switched to
            # Add a short delay to ensure the window is fully mounted and ready
            def populate_ccp_widgets():
                try:
                    # Check if the window is actually initialized
                    ccp_window = self.query_one("#conversations_characters_prompts-window")
                    if isinstance(ccp_window, PlaceholderWindow):
                        # Window isn't initialized yet, skip population
                        loguru_logger.warning("CCP window is still a placeholder, skipping widget population")
                        return
                    
                    # Now it's safe to populate widgets
                    self.call_after_refresh(ccp_handlers.populate_ccp_character_select, self)
                    self.call_after_refresh(ccp_handlers.populate_ccp_prompts_list_view, self)
                    self.call_after_refresh(ccp_handlers.populate_ccp_dictionary_select, self)
                    self.call_after_refresh(ccp_handlers.populate_ccp_worldbook_list, self)
                    self.call_after_refresh(ccp_handlers.perform_ccp_conversation_search, self)
                except QueryError:
                    loguru_logger.error("CCP window not found during widget population")
            
            # Use a timer to ensure the window is fully initialized
            self.set_timer(0.1, populate_ccp_widgets)
        elif new_tab == TAB_NOTES:
            # Use call_after_refresh for async function
            self.call_after_refresh(notes_handlers.load_and_display_notes_handler, self)
        elif new_tab == TAB_MEDIA:
            def activate_media_initial_view():
                try:
                    media_window = self.query_one(MediaWindow)
                    media_window.activate_initial_view()
                except QueryError:
                    loguru_logger.error("Could not find MediaWindow to activate its initial view.")
            
            # Use a timer to ensure the window is fully initialized
            self.set_timer(0.1, activate_media_initial_view)
        elif new_tab == TAB_SEARCH:
            # Handle search tab initialization with a delay to ensure window is ready
            def initialize_search_tab():
                try:
                    # Check if the window is actually initialized
                    search_window = self.query_one("#search-window")
                    if isinstance(search_window, PlaceholderWindow):
                        # Window isn't initialized yet, skip setting sub-tab
                        loguru_logger.warning("Search window is still a placeholder, skipping sub-tab initialization")
                        return
                    
                    # Now it's safe to set the active sub-tab
                    if not self.search_active_sub_tab:
                        self.search_active_sub_tab = self._initial_search_sub_tab_view
                except QueryError:
                    loguru_logger.error("Search window not found during initialization")
            
            # Use a timer to ensure the window is fully initialized
            self.set_timer(0.1, initialize_search_tab)
        elif new_tab == TAB_INGEST:
            if not self.ingest_active_view:
                self.loguru_logger.debug(
                    f"Switched to Ingest tab, activating initial view: {self._initial_ingest_view}") # Reverted to original debug log
                # Use call_later to ensure the UI has settled after tab switch before changing sub-view
                self.call_later(self._activate_initial_ingest_view)
        elif new_tab == TAB_TOOLS_SETTINGS:
            # Handle tools settings tab initialization with proper placeholder check
            def initialize_tools_settings():
                try:
                    # Check if the window is actually initialized
                    tools_window = self.query_one("#tools_settings-window")
                    if isinstance(tools_window, PlaceholderWindow):
                        # Window isn't initialized yet, try again later silently
                        self.set_timer(0.1, initialize_tools_settings)
                        return
                    
                    # Now it's safe to activate the initial view
                    from .UI.Tools_Settings_Window import ToolsSettingsWindow
                    if isinstance(tools_window, ToolsSettingsWindow):
                        tools_window.activate_initial_view()
                        if not self.tools_settings_active_view:
                            self.tools_settings_active_view = self._initial_tools_settings_view
                            self.loguru_logger.debug(f"Tools & Settings tab initialized with view: {self._initial_tools_settings_view}")
                except QueryError:
                    self.loguru_logger.error("Tools settings window not found during initialization")
            
            # Use a timer to ensure the window is ready
            self.set_timer(0.1, initialize_tools_settings)
        elif new_tab == TAB_LLM:  # New elif block for LLM tab
            if not self.llm_active_view:  # If no view is active yet
                self.loguru_logger.debug(
                    f"Switched to LLM Management tab, activating initial view: {self._initial_llm_view}")
                self.call_later(setattr, self, 'llm_active_view', self._initial_llm_view)
            # Populate LLM help texts when the tab is shown
            self.call_after_refresh(llm_management_events.populate_llm_help_texts, self)
        elif new_tab == TAB_EVALS: # Added for Evals tab
            # EvalsLab is a unified dashboard - no need for view activation
            self.loguru_logger.debug(f"Switched to Evals tab")


    def _log_view_dimensions(self, view, parent):
        """Helper to log view dimensions after refresh."""
        self.loguru_logger.info(f"After refresh - View {view.id} dimensions: width={view.size.width}, height={view.size.height}")
        self.loguru_logger.info(f"After refresh - Parent dimensions: width={parent.size.width}, height={parent.size.height}")
    
    async def _activate_initial_ingest_view(self) -> None:
        self.loguru_logger.info("Attempting to activate initial ingest view via _activate_initial_ingest_view.")
        
        # First, ensure all views are hidden initially
        try:
            content_pane = self.query_one("#ingest-content-pane")
            for child in content_pane.children:
                if child.id and child.id.startswith("ingest-view-"):
                    child.styles.display = "none"
                    self.loguru_logger.debug(f"Initially hiding ingest view: {child.id}")
        except QueryError:
            self.loguru_logger.error("Could not find #ingest-content-pane to hide views initially")
        
        if not self.ingest_active_view: # Check if it hasn't been set by some other means already
            self.loguru_logger.debug(f"Setting ingest_active_view to initial: {self._initial_ingest_view}")
            self.ingest_active_view = self._initial_ingest_view
        else:
            self.loguru_logger.debug(f"Ingest active view already set to '{self.ingest_active_view}'. No change made by _activate_initial_ingest_view.")


    # Watchers for sidebar collapsed states (keep as is)
    def watch_chat_sidebar_collapsed(self, collapsed: bool) -> None:
        if not self._ui_ready: # Keep the UI ready guard
            self.loguru_logger.debug("watch_chat_sidebar_collapsed: UI not ready.")
            return
        try:
            # Query for the new ID
            sidebar = self.query_one("#chat-left-sidebar") # <<< CHANGE THIS LINE
            sidebar.display = not collapsed # True = visible, False = hidden
            self.loguru_logger.debug(f"Chat left sidebar (#chat-left-sidebar) display set to {not collapsed}")
        except QueryError:
            # Update the error message to reflect the new ID
            self.loguru_logger.error("Chat left sidebar (#chat-left-sidebar) not found by watcher.") # <<< UPDATE ERROR MSG
        except Exception as e:
            self.loguru_logger.error(f"Error toggling chat left sidebar: {e}", exc_info=True)

    def watch_chat_right_sidebar_collapsed(self, collapsed: bool) -> None:
        """Hide or show the character settings sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#chat-right-sidebar")  # ID from create_chat_right_sidebar
            sidebar.display = not collapsed
        except QueryError:
            logging.error("Character sidebar widget (#chat-right-sidebar) not found.")
    
    def watch_chat_right_sidebar_width(self, width: int) -> None:
        """Update the width of the chat right sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#chat-right-sidebar", VerticalScroll)
            sidebar.styles.width = f"{width}%"
        except QueryError:
            # Sidebar might not be created yet
            pass

    def watch_notes_sidebar_left_collapsed(self, collapsed: bool) -> None:
        """Hide or show the notes left sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#notes-sidebar-left", NotesSidebarLeft)
            sidebar.display = not collapsed
            # Optional: adjust layout of notes-main-content if needed
        except QueryError:
            logging.error("Notes left sidebar widget (#notes-sidebar-left) not found.")

    def watch_notes_sidebar_right_collapsed(self, collapsed: bool) -> None:
        """Hide or show the notes right sidebar."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar = self.query_one("#notes-sidebar-right", NotesSidebarRight)
            sidebar.display = not collapsed
            # Optional: adjust layout of notes-main-content if needed
        except QueryError:
            logging.error("Notes right sidebar widget (#notes-sidebar-right) not found.")
    
    def watch_notes_unsaved_changes(self, has_unsaved: bool) -> None:
        """Update the unsaved changes indicator."""
        if not self._ui_ready:
            return
        try:
            indicator = self.query_one("#notes-unsaved-indicator", Label)
            # Don't update if we're showing auto-save status
            if self.notes_auto_save_status:
                return
            if has_unsaved:
                indicator.update("● Unsaved")
                indicator.add_class("has-unsaved")
            else:
                indicator.update("")
                indicator.remove_class("has-unsaved")
        except QueryError:
            pass  # Indicator might not exist yet

    def watch_notes_auto_save_status(self, status: str) -> None:
        """Update the indicator based on auto-save status."""
        if not self._ui_ready:
            return
        try:
            indicator = self.query_one("#notes-unsaved-indicator", Label)
            if status == "saving":
                indicator.update("⟳ Auto-saving...")
                indicator.remove_class("has-unsaved")
                indicator.add_class("auto-saving")
            elif status == "saved":
                indicator.update("✓ Saved")
                indicator.remove_class("has-unsaved", "auto-saving")
                indicator.add_class("saved")
                # Clear the saved status after 2 seconds
                self.set_timer(2.0, lambda: setattr(self, 'notes_auto_save_status', ''))
            else:
                # Empty status - let the unsaved changes watcher handle it
                indicator.remove_class("auto-saving", "saved")
                # Re-evaluate unsaved changes
                if self.notes_unsaved_changes:
                    indicator.update("● Unsaved")
                    indicator.add_class("has-unsaved")
                else:
                    indicator.update("")
                    indicator.remove_class("has-unsaved")
        except QueryError:
            pass  # Indicator might not exist yet

    def watch_conv_char_sidebar_left_collapsed(self, collapsed: bool) -> None:
        """Hide or show the Conversations, Characters & Prompts left sidebar pane."""
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        try:
            sidebar_pane = self.query_one("#conv-char-left-pane") # The ID of the VerticalScroll
            sidebar_pane.display = not collapsed # True means visible, False means hidden
            logging.debug(f"Conversations, Characters & Prompts left pane display set to {not collapsed}")
        except QueryError:
            logging.error("Conversations, Characters & Prompts left sidebar pane (#conv-char-left-pane) not found.")
        except Exception as e:
            logging.error(f"Error toggling Conversations, Characters & Prompts left sidebar pane: {e}", exc_info=True)

    def watch_evals_sidebar_collapsed(self, collapsed: bool) -> None:
        """EvalsLab uses unified dashboard - no sidebar to collapse."""
        # This method is kept for backwards compatibility but does nothing
        # The new EvalsLab UI doesn't have a collapsible sidebar
        pass
    
    def watch_media_active_view(self, old_view: Optional[str], new_view: Optional[str]) -> None:
        """Notify MediaWindow when media_active_view changes."""
        # Temporarily disabled - MediaWindow handles its own navigation via MediaTypeSelectedEvent
        pass
        # if not self._ui_ready:
        #     self.loguru_logger.debug("watch_media_active_view: UI not ready.")
        #     return
        # 
        # if self.current_tab == TAB_MEDIA:
        #     try:
        #         media_window = self.query_one(MediaWindow)
        #         # Sync the MediaWindow's own media_active_view
        #         media_window.media_active_view = new_view
        #         # Call the watcher manually to trigger the view change
        #         if new_view:
        #             media_window.watch_media_active_view(old_view, new_view)
        #         self.loguru_logger.info(f"Notified MediaWindow of view change: {old_view} -> {new_view}")
        #     except QueryError:
        #         self.loguru_logger.error("MediaWindow not found for view update.")
        #     except Exception as e:
        #         self.loguru_logger.error(f"Error updating MediaWindow view: {e}", exc_info=True)

    def show_ingest_view(self, view_id_to_show: Optional[str]):
        """
        Shows the specified ingest view within the ingest-content-pane and hides others.
        If view_id_to_show is None, hides all ingest views.
        """
        self.log.debug(f"Attempting to show ingest view: {view_id_to_show}")
        try:
            ingest_content_pane = self.query_one("#ingest-content-pane")
            if view_id_to_show:
                ingest_content_pane.display = True
        except QueryError:
            self.log.error("#ingest-content-pane not found. Cannot manage ingest views.")
            return

        for view_id in self.ALL_INGEST_VIEW_IDS:
            try:
                view_container = self.query_one(f"#{view_id}")
                is_target = (view_id == view_id_to_show)
                view_container.display = is_target
                if is_target:
                    self.log.info(f"Displaying ingest view: #{view_id}")
                    # Initialize models for video/audio windows when they become visible
                    if view_id == "ingest-view-local-video":
                        self._initialize_video_models()
                    elif view_id == "ingest-view-local-audio":
                        self._initialize_audio_models()
            except QueryError:
                self.log.warning(f"Ingest view container '#{view_id}' not found during show_ingest_view.")

    def _initialize_video_models(self) -> None:
        """Initialize models for the video ingestion window."""
        try:
            ingest_window = self.query_one("#ingest-window", NewIngestWindow)
            # New ingest window doesn't need model initialization
            self.log.debug("New ingest window loaded")
        except Exception as e:
            self.log.debug(f"Could not initialize video models: {e}")

    def _initialize_audio_models(self) -> None:
        """Initialize models for the audio ingestion window."""
        try:
            ingest_window = self.query_one("#ingest-window", NewIngestWindow)
            # New ingest window doesn't need model initialization
            self.log.debug("New ingest window loaded")
        except Exception as e:
            self.log.debug(f"Could not initialize audio models: {e}")

    async def save_current_note(self) -> bool:
        """Saves the currently selected note's title and content to the database."""
        if not self.notes_service or not self.current_selected_note_id or self.current_selected_note_version is None:
            logging.warning("No note selected or service unavailable. Cannot save.")
            # Optionally: self.notify("No note selected to save.", severity="warning")
            return False

        try:
            editor = self.query_one("#notes-editor-area", TextArea)
            title_input = self.query_one("#notes-title-input", Input)
            current_content = editor.text
            current_title = title_input.value

            # Check if title or content actually changed to avoid unnecessary saves.
            # This requires storing the original loaded title/content if not already done by reactives.
            # For now, we save unconditionally if a note is selected.
            # A more advanced check could compare with self.current_selected_note_title and self.current_selected_note_content

            logging.info(
                f"Attempting to save note ID: {self.current_selected_note_id}, Version: {self.current_selected_note_version}")
            success = self.notes_service.update_note(
                user_id=self.notes_user_id,
                note_id=self.current_selected_note_id,
                update_data={'title': current_title, 'content': current_content},
                expected_version=self.current_selected_note_version
            )
            if success:
                logging.info(f"Note {self.current_selected_note_id} saved successfully.")
                # Update version and potentially title/content reactive vars if update_note returns new state
                # For now, we re-fetch to get the new version.
                updated_note_details = self.notes_service.get_note_by_id(
                    user_id=self.notes_user_id,
                    note_id=self.current_selected_note_id
                )
                if updated_note_details:
                    self.current_selected_note_version = updated_note_details.get('version')
                    self.current_selected_note_title = updated_note_details.get('title')  # Update reactive
                    # self.current_selected_note_content = updated_note_details.get('content') # Update reactive

                    # Refresh the list in the left sidebar to reflect title changes and update item version
                    await notes_handlers.load_and_display_notes_handler(self)
                    # self.notify("Note saved!", severity="information") # If notifications are setup
                else:
                    # Note might have been deleted by another client after our successful update, though unlikely.
                    logging.warning(f"Note {self.current_selected_note_id} not found after presumably successful save.")
                    # self.notify("Note saved, but failed to refresh details.", severity="warning")

                return True
            else:
                # This path should ideally not be reached if update_note raises exceptions on failure.
                logging.warning(
                    f"notes_service.update_note for {self.current_selected_note_id} returned False without raising error.")
                # self.notify("Failed to save note (unknown reason).", severity="error")
                return False

        except ConflictError as e:
            logging.error(f"Conflict saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify(f"Save conflict: {e}. Please reload the note.", severity="error")
            # Optionally, offer to reload the note or overwrite. For now, just log.
            # await self.handle_save_conflict() # A new method to manage this
            return False
        except CharactersRAGDBError as e:
            logging.error(f"Database error saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify("Error saving note to database.", severity="error")
            return False
        except QueryError as e:
            logging.error(f"UI component not found while saving note: {e}", exc_info=True)
            # self.notify("UI error while saving note.", severity="error")
            return False
        except Exception as e:
            logging.error(f"Unexpected error saving note {self.current_selected_note_id}: {e}", exc_info=True)
            # self.notify("Unexpected error saving note.", severity="error")
            return False


    #######################################################################
    # --- Notes UI Event Handlers (Chat Tab Sidebar) ---
    #######################################################################
    @on(Button.Pressed, "#chat-notes-create-new-button")
    async def handle_chat_notes_create_new(self, event: Button.Pressed) -> None:
        """Handles the 'Create New Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Attempting to create new note for user: {self.notes_user_id}")
        default_title = "New Note"
        default_content = ""

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_create_new.")
            return

        try:
            # 1. Call self.notes_service.add_note
            new_note_id = self.notes_service.add_note(
                user_id=self.notes_user_id,
                title=default_title,
                content=default_content,
                # keywords, parent_id, etc., can be added if needed
            )

            if new_note_id:
                # 2. Store Note ID and Version
                self.current_chat_note_id = new_note_id
                self.current_chat_note_version = 1  # Assuming version starts at 1
                self.loguru_logger.info(f"New note created with ID: {new_note_id}, Version: {self.current_chat_note_version}")

                # 3. Update UI Input Fields
                title_input = self.query_one("#chat-notes-title-input", Input)
                content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)
                title_input.value = default_title
                content_textarea.text = default_content

                # 4. Add to ListView
                notes_list_view = self.query_one("#chat-notes-listview", ListView)
                new_list_item = ListItem(Label(default_title))
                new_list_item.id = f"note-item-{new_note_id}" # Ensure unique DOM ID for the ListItem itself
                # We'll need a way to store the actual note_id on the ListItem for retrieval,
                # Textual's ListItem doesn't have a direct `data` attribute.
                # A common pattern is to use a custom ListItem subclass or manage a mapping.
                # For now, we can set the DOM ID and parse it, or use a custom attribute if we make one.
                # setattr(new_list_item, "_note_id", new_note_id) # Example of custom attribute
                # Or, more simply for now, we can rely on on_chat_notes_collapsible_toggle to refresh the whole list
                # which will then pick up the new note from the DB.
                # For immediate feedback without full list refresh:
                # ListView doesn't have prepend, so we'll append and let the list refresh handle ordering
                await notes_list_view.append(new_list_item)

                # 5. Set Focus
                title_input.focus()

                self.notify("New note created in sidebar.", severity="information")
            else:
                self.notify("Failed to create new note (no ID returned).", severity="error")
                self.loguru_logger.error("notes_service.add_note did not return a new_note_id.")

        except CharactersRAGDBError as e: # Specific DB error
            self.loguru_logger.error(f"Database error creating new note: {e}", exc_info=True)
            self.notify(f"DB error creating note: {e}", severity="error")
        except Exception as e: # Catch-all for other unexpected errors
            self.loguru_logger.error(f"Unexpected error creating new note: {e}", exc_info=True)
            self.notify(f"Error creating note: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-search-button")
    async def handle_chat_notes_search(self, event: Button.Pressed) -> None:
        """Handles the 'Search' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Search Notes button pressed. User ID: {self.notes_user_id}")

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_search.")
            return

        try:
            search_input = self.query_one("#chat-notes-search-input", Input)
            search_term = search_input.value.strip()

            notes_list_view = self.query_one("#chat-notes-listview", ListView)
            await notes_list_view.clear()

            listed_notes: List[Dict[str, Any]] = []
            limit = 50

            if not search_term:
                self.loguru_logger.info("Empty search term, listing all notes.")
                listed_notes = self.notes_service.list_notes(user_id=self.notes_user_id, limit=limit)
            else:
                self.loguru_logger.info(f"Searching notes with term: '{search_term}'")
                listed_notes = self.notes_service.search_notes(user_id=self.notes_user_id, search_term=search_term, limit=limit)

            if listed_notes:
                for note in listed_notes:
                    note_title = note.get('title', 'Untitled Note')
                    note_id = note.get('id')
                    if not note_id:
                        self.loguru_logger.warning(f"Note found without an ID during search: {note_title}. Skipping.")
                        continue

                    list_item_label = Label(note_title)
                    new_list_item = ListItem(list_item_label)
                    new_list_item.id = f"note-item-{note_id}"
                    # setattr(new_list_item, "_note_data", note) # If needed later for load
                    await notes_list_view.append(new_list_item)

                self.notify(f"{len(listed_notes)} notes found.", severity="information")
                self.loguru_logger.info(f"Populated notes list with {len(listed_notes)} search results.")
                self.loguru_logger.debug(f"ListView child count after search population: {len(notes_list_view.children)}") # Fixed: use len(children) instead of child_count
            else:
                msg = "No notes match your search." if search_term else "No notes found."
                self.notify(msg, severity="information")
                self.loguru_logger.info(msg)

        except CharactersRAGDBError as e:
            self.loguru_logger.error(f"Database error searching notes: {e}", exc_info=True)
            self.notify(f"DB error searching notes: {e}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.error(f"UI element not found during notes search: {e_query}", exc_info=True)
            self.notify("UI error during notes search.", severity="error")
        except Exception as e:
            self.loguru_logger.error(f"Unexpected error searching notes: {e}", exc_info=True)
            self.notify(f"Error searching notes: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-load-button")
    async def handle_chat_notes_load(self, event: Button.Pressed) -> None:
        """Handles the 'Load Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Load Note button pressed. User ID: {self.notes_user_id}")

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_load.")
            return

        try:
            notes_list_view = self.query_one("#chat-notes-listview", ListView)
            selected_list_item = notes_list_view.highlighted_child

            if selected_list_item is None or not selected_list_item.id:
                self.notify("Please select a note to load.", severity="warning")
                return

            # Extract actual_note_id from the ListItem's DOM ID
            dom_id_parts = selected_list_item.id.split("note-item-")
            if len(dom_id_parts) < 2 or not dom_id_parts[1]:
                self.notify("Selected item has an invalid ID format.", severity="error")
                self.loguru_logger.error(f"Invalid ListItem ID format: {selected_list_item.id}")
                return

            actual_note_id = dom_id_parts[1]
            self.loguru_logger.info(f"Attempting to load note with ID: {actual_note_id}")

            note_data = self.notes_service.get_note_by_id(user_id=self.notes_user_id, note_id=actual_note_id)

            if note_data:
                title_input = self.query_one("#chat-notes-title-input", Input)
                content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)

                loaded_title = note_data.get('title', '')
                loaded_content = note_data.get('content', '')
                loaded_version = note_data.get('version')
                loaded_id = note_data.get('id')

                title_input.value = loaded_title
                content_textarea.text = loaded_content

                self.current_chat_note_id = loaded_id
                self.current_chat_note_version = loaded_version

                self.notify(f"Note '{loaded_title}' loaded.", severity="information")
                self.loguru_logger.info(f"Note '{loaded_title}' (ID: {loaded_id}, Version: {loaded_version}) loaded into UI.")
            else:
                self.notify(f"Could not load note (ID: {actual_note_id}). It might have been deleted.", severity="warning")
                self.loguru_logger.warning(f"Note with ID {actual_note_id} not found by service.")
                # Clear fields if note not found
                self.query_one("#chat-notes-title-input", Input).value = ""
                self.query_one("#chat-notes-content-textarea", TextArea).text = ""
                self.current_chat_note_id = None
                self.current_chat_note_version = None

        except CharactersRAGDBError as e_db:
            self.loguru_logger.error(f"Database error loading note: {e_db}", exc_info=True)
            self.notify(f"DB error loading note: {e_db}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.error(f"UI element not found during note load: {e_query}", exc_info=True)
            self.notify("UI error during note load.", severity="error")
        except Exception as e:
            self.loguru_logger.error(f"Unexpected error loading note: {e}", exc_info=True)
            self.notify(f"Error loading note: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-save-button")
    async def handle_chat_notes_save(self, event: Button.Pressed) -> None:
        """Handles the 'Save Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info(f"Save Note button pressed. User ID: {self.notes_user_id}")

        if not self.notes_service:
            self.notify("Notes service is not available.", severity="error")
            self.loguru_logger.error("Notes service not available in handle_chat_notes_save.")
            return

        if not self.current_chat_note_id or self.current_chat_note_version is None:
            self.notify("No active note to save. Load or create a note first.", severity="warning")
            self.loguru_logger.warning("handle_chat_notes_save called without an active note_id or version.")
            return

        try:
            title_input = self.query_one("#chat-notes-title-input", Input)
            content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)

            title = title_input.value
            content = content_textarea.text

            update_data = {"title": title, "content": content}

            self.loguru_logger.info(f"Attempting to save note ID: {self.current_chat_note_id}, Version: {self.current_chat_note_version}")

            success = self.notes_service.update_note(
                user_id=self.notes_user_id,
                note_id=self.current_chat_note_id,
                update_data=update_data,
                expected_version=self.current_chat_note_version
            )

            if success: # Should be true if no exception was raised by DB layer for non-Conflict errors
                self.current_chat_note_version += 1
                self.notify("Note saved successfully.", severity="information")
                self.loguru_logger.info(f"Note {self.current_chat_note_id} saved. New version: {self.current_chat_note_version}")

                # Update ListView item
                try:
                    notes_list_view = self.query_one("#chat-notes-listview", ListView)
                    # Find the specific ListItem to update its Label
                    # This requires iterating or querying if the ListItem's DOM ID is known
                    item_dom_id = f"note-item-{self.current_chat_note_id}"
                    for item in notes_list_view.children:
                        if isinstance(item, ListItem) and item.id == item_dom_id:
                            # Assuming the first child of ListItem is the Label we want to update
                            label_to_update = item.query_one(Label)
                            label_to_update.update(title) # Update with the new title
                            self.loguru_logger.debug(f"Updated title in ListView for note ID {self.current_chat_note_id} to '{title}'")
                            break
                        else:
                            self.loguru_logger.debug(f"ListItem with ID {item_dom_id} not found for title update after save (iterated item ID: {item.id}).")
                except QueryError as e_lv_update:
                    self.loguru_logger.error(f"Error querying Label within ListView item to update title: {e_lv_update}")
                except Exception as e_item_update: # Catch other errors during list item update
                    self.loguru_logger.error(f"Unexpected error updating list item title: {e_item_update}", exc_info=True)
            else:
                # This case might not be hit if service raises exceptions for all failures
                self.notify("Failed to save note. Reason unknown.", severity="error")
                self.loguru_logger.error(f"notes_service.update_note returned False for note {self.current_chat_note_id}")

        except ConflictError:
            self.loguru_logger.warning(f"Save conflict for note {self.current_chat_note_id}. Expected version: {self.current_chat_note_version}")
            self.notify("Save conflict: Note was modified elsewhere. Please reload and reapply changes.", severity="error", timeout=10)
        except CharactersRAGDBError as e_db:
            self.loguru_logger.error(f"Database error saving note {self.current_chat_note_id}: {e_db}", exc_info=True)
            self.notify(f"DB error saving note: {e_db}", severity="error")
        except QueryError as e_query:
            self.loguru_logger.error(f"UI element not found during note save: {e_query}", exc_info=True)
            self.notify("UI error during note save.", severity="error")
        except Exception as e:
            self.loguru_logger.error(f"Unexpected error saving note {self.current_chat_note_id}: {e}", exc_info=True)
            self.notify(f"Error saving note: {type(e).__name__}", severity="error")

    @on(Button.Pressed, "#chat-notes-copy-button")
    async def handle_chat_notes_copy(self, event: Button.Pressed) -> None:
        """Handles the 'Copy Note' button press in the chat sidebar's notes section."""
        self.loguru_logger.info("Copy Note button pressed.")
        
        try:
            # Get title and content from the input fields
            title_input = self.query_one("#chat-notes-title-input", Input)
            content_textarea = self.query_one("#chat-notes-content-textarea", TextArea)
            
            title = title_input.value.strip()
            content = content_textarea.text.strip()
            
            # Check if there's anything to copy
            if not title and not content:
                self.notify("No note content to copy.", severity="warning")
                return
            
            # Format the note for clipboard
            if title and content:
                # Both title and content present
                formatted_note = f"# {title}\n\n{content}"
            elif title:
                # Only title
                formatted_note = f"# {title}"
            else:
                # Only content
                formatted_note = content
            
            # Copy to clipboard
            self.copy_to_clipboard(formatted_note)
            self.notify("Note copied to clipboard!", severity="information")
            self.loguru_logger.info(f"Note copied to clipboard. Title: '{title[:50]}...'" if title else "Note content copied to clipboard.")
            
        except QueryError as e:
            self.loguru_logger.error(f"UI element not found during note copy: {e}")
            self.notify("UI error during note copy.", severity="error")
        except Exception as e:
            self.loguru_logger.error(f"Unexpected error copying note: {e}", exc_info=True)
            self.notify(f"Error copying note: {type(e).__name__}", severity="error")

    @on(Collapsible.Toggled, "#chat-notes-collapsible")
    async def on_chat_notes_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Notes collapsible section in the chat sidebar."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info(f"Notes collapsible opened in chat sidebar. User ID: {self.notes_user_id}. Refreshing list.")

            if not self.notes_service:
                self.notify("Notes service is not available.", severity="error")
                self.loguru_logger.error("Notes service not available in on_chat_notes_collapsible_toggle.")
                return

    @on(Collapsible.Toggled, "#chat-active-character-info-collapsible")
    async def on_chat_active_character_info_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Active Character Info collapsible section in the chat sidebar."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Active Character Info collapsible opened in chat sidebar. Refreshing character list.")

            # Call the function to populate the character list
            from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
            await chat_events._populate_chat_character_search_list(self)

            try:
                # 1. Clear ListView
                notes_list_view = self.query_one("#chat-notes-listview", ListView)
                await notes_list_view.clear()

                # 2. Call self.notes_service.list_notes
                # Limit to a reasonable number, e.g., 50, most recent first if service supports sorting
                listed_notes = self.notes_service.list_notes(user_id=self.notes_user_id, limit=50)

                # 3. Populate ListView
                if listed_notes:
                    for note in listed_notes:
                        note_title = note.get('title', 'Untitled Note')
                        note_id = note.get('id')
                        if not note_id:
                            self.loguru_logger.warning(f"Note found without an ID: {note_title}. Skipping.")
                            continue

                        list_item_label = Label(note_title)
                        new_list_item = ListItem(list_item_label)
                        # Store the actual note_id on the ListItem for retrieval.
                        # Using a unique DOM ID for the ListItem itself.
                        new_list_item.id = f"note-item-{note_id}"
                        # A custom attribute to store data:
                        # setattr(new_list_item, "_note_data", note) # Store whole note or just id/version

                        await notes_list_view.append(new_list_item)
                    self.notify("Notes list refreshed.", severity="information")
                    self.loguru_logger.info(f"Populated notes list with {len(listed_notes)} items.")
                else:
                    self.notify("No notes found.", severity="information")
                    self.loguru_logger.info("No notes found for user after refresh.")

            except CharactersRAGDBError as e: # Specific DB error
                self.loguru_logger.error(f"Database error listing notes: {e}", exc_info=True)
                self.notify(f"DB error listing notes: {e}", severity="error")
            except QueryError as e_query: # If UI elements are not found
                 self.loguru_logger.error(f"UI element not found in notes toggle: {e_query}", exc_info=True)
                 self.notify("UI error while refreshing notes.", severity="error")
            except Exception as e: # Catch-all for other unexpected errors
                self.loguru_logger.error(f"Unexpected error listing notes: {e}", exc_info=True)
                self.notify(f"Error listing notes: {type(e).__name__}", severity="error")
        else:
            self.loguru_logger.info("Notes collapsible closed in chat sidebar.")

    @on(Collapsible.Toggled, "#chat-active-character-info-collapsible")
    async def on_chat_active_character_info_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Active Character Info collapsible section in the chat sidebar."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Active Character Info collapsible opened in chat sidebar. Refreshing character list.")

            # Call the function to populate the character list
            from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
            await chat_events._populate_chat_character_search_list(self)
        else:
            self.loguru_logger.info("Active Character Info collapsible closed in chat sidebar.")

    @on(Collapsible.Toggled, "#chat-conversations")
    async def on_chat_conversations_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Conversations collapsible section in the chat sidebar."""
        # Check if this is specifically the chat conversations collapsible
        if event.collapsible.id != "chat-conversations":
            return
            
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Conversations collapsible opened in chat sidebar.")
            
            # Populate the character filter dropdown only once when the collapsible is first opened
            # This avoids the database connection conflicts that occur during startup
            if not self._chat_character_filter_populated:
                self.loguru_logger.info("Populating character filter for the first time.")
                try:
                    from tldw_chatbook.Event_Handlers.Chat_Events import chat_events
                    await chat_events.populate_chat_conversation_character_filter_select(self)
                    self._chat_character_filter_populated = True
                    self.loguru_logger.info("Character filter populated successfully.")
                except Exception as e:
                    self.loguru_logger.error(f"Failed to populate character filter: {e}", exc_info=True)
            else:
                self.loguru_logger.debug("Character filter already populated, skipping.")
        else:
            self.loguru_logger.info("Conversations collapsible closed in chat sidebar.")
    
    @on(Collapsible.Toggled, "#conv-char-conversations-collapsible")
    async def on_ccp_conversations_collapsible_toggle(self, event: Collapsible.Toggled) -> None:
        """Handles the expansion/collapse of the Conversations collapsible section in the CCP tab."""
        if not event.collapsible.collapsed:  # If the collapsible was just expanded
            self.loguru_logger.info("Conversations collapsible opened in CCP tab.")
            # Trigger initial search to populate the list
            await ccp_handlers.perform_ccp_conversation_search(self)
        else:
            self.loguru_logger.info("Conversations collapsible closed in CCP tab.")

    ########################################################################
    #
    # --- EVENT DISPATCHERS ---
    #
    ########################################################################
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dispatches button presses to the appropriate event handler using a map."""
        button_id = event.button.id
        if not button_id:
            return

        self.loguru_logger.info(f"Button pressed: ID='{button_id}' on Tab='{self.current_tab}'")

        # 1. Handle global tab switching first
        if button_id.startswith("tab-"):
            await tab_events.handle_tab_button_pressed(self, event)
            return

        # 2. Try to delegate to the appropriate window component
        try:
            # Determine which window component should handle this button press based on current tab
            window_id_map = {
                TAB_CHAT: "chat-window",
                TAB_CCP: "conversations_characters_prompts-window",
                TAB_NOTES: "notes-window",
                TAB_MEDIA: "media-window",
                TAB_SEARCH: "search-window",
                TAB_INGEST: "ingest-window",
                TAB_TOOLS_SETTINGS: "tools_settings-window",
                TAB_LLM: "llm_management-window",
                TAB_CUSTOMIZE: "customize-window",
                TAB_LOGS: "logs-window",
                TAB_STATS: "stats-window",
                TAB_EVALS: "evals-window",
                TAB_CODING: "coding-window",
                TAB_STTS: "stts-window",
                TAB_STUDY: "study-window",
                TAB_CHATBOOKS: "chatbooks-window"
            }

            window_id = window_id_map.get(self.current_tab)
            self.loguru_logger.info(f"Window ID for tab '{self.current_tab}': {window_id}")
            if window_id:
                window = self.query_one(f"#{window_id}")
                self.loguru_logger.info(f"Found window: {type(window).__name__}")
                # Check if the window has an on_button_pressed method
                has_method = hasattr(window, "on_button_pressed") and callable(window.on_button_pressed)
                self.loguru_logger.info(f"Window has on_button_pressed: {has_method}")
                if has_method:
                    # Call the window's button handler - it might be async
                    self.loguru_logger.info(f"Delegating to window's on_button_pressed")
                    result = window.on_button_pressed(event)
                    if inspect.isawaitable(result):
                        await result
                    # Check if event has been stopped (some event types don't have is_stopped)
                    if hasattr(event, 'is_stopped') and event.is_stopped:
                        self.loguru_logger.info(f"Event was stopped by window handler")
                        return
                    self.loguru_logger.info(f"Window handler completed, event not stopped")
                    # Don't return here - let it fall through to the handler map!

        except QueryError:
            self.loguru_logger.error(f"Could not find window component for tab '{self.current_tab}'")
        except Exception as e:
            self.loguru_logger.error(f"Error delegating button press to window component: {e}", exc_info=True)

        # 3. Use the handler map for buttons not handled by window components
        current_tab_handlers = self.button_handler_map.get(self.current_tab, {})
        handler = current_tab_handlers.get(button_id)
        
        self.loguru_logger.info(f"Looking for handler for button '{button_id}' in tab '{self.current_tab}'")
        self.loguru_logger.info(f"Available handlers for this tab: {list(current_tab_handlers.keys())}")
        self.loguru_logger.info(f"Handler found: {handler is not None}")
        
        # Special debug logging for save chat button
        if button_id == "chat-save-current-chat-button":
            self.loguru_logger.info(f"Save Temp Chat button pressed - Handler found: {handler is not None}")
            self.loguru_logger.info(f"Current tab: {self.current_tab}, Expected: {TAB_CHAT}")

        if handler:
            if callable(handler):
                try:
                    # Call the handler, which is expected to return a coroutine (an awaitable object).
                    result = handler(self, event)

                    # Check if the result is indeed awaitable before awaiting it.
                    # This makes the code more robust and satisfies static type checkers.
                    if inspect.isawaitable(result):
                        await result
                    else:
                        self.loguru_logger.warning(
                            f"Handler for button '{button_id}' did not return an awaitable object."
                        )
                except Exception as e:
                    self.loguru_logger.error(f"Error executing handler for button '{button_id}': {e}", exc_info=True)
                    self.notify(f"Error handling button action: {str(e)[:100]}", severity="error")
            else:
                self.loguru_logger.error(f"Handler for button '{button_id}' is not callable: {handler}")
            return  # The button press was handled (or an error occurred).

        # 4. Fallback for unmapped buttons
        self.loguru_logger.warning(f"Unhandled button press for ID '{button_id}' on tab '{self.current_tab}'.")

    async def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handles text area changes, e.g., for live updates to character data."""
        control_id = event.control.id
        current_active_tab = self.current_tab

        if current_active_tab == TAB_CHAT and control_id and control_id.startswith("chat-character-"):
            # Ensure it's one of the actual attribute TextAreas, not something else
            if control_id in [
                "chat-character-description-edit",
                "chat-character-personality-edit",
                "chat-character-scenario-edit",
                "chat-character-system-prompt-edit",
                "chat-character-first-message-edit"
            ]:
                await chat_handlers.handle_chat_character_attribute_changed(self, event)
        elif current_active_tab == TAB_NOTES and control_id == "notes-editor-area":
            # Handle notes editor changes
            await notes_handlers.handle_notes_editor_changed(self, event)

    def _update_model_download_log(self, message: str) -> None:
        """Helper to write messages to the model download log widget."""
        LogWidgetManager.update_model_download_log(self, message)

    def _update_mlx_log(self, message: str) -> None:
        """Helper to write messages to the MLX-LM log widget."""
        LogWidgetManager.update_mlx_log(self, message)

    async def on_input_changed(self, event: Input.Changed) -> None:
        input_id = event.input.id
        current_active_tab = self.current_tab
        # --- Notes Search ---
        if input_id == "notes-search-input" and current_active_tab == TAB_NOTES: # Changed from elif to if
            await notes_handlers.handle_notes_search_input_changed(self, event.value)
        elif input_id == "notes-keyword-filter-input" and current_active_tab == TAB_NOTES:
            await notes_handlers.handle_notes_keyword_filter_input_changed(self, event.value)
        elif input_id == "notes-title-input" and current_active_tab == TAB_NOTES:
            await notes_handlers.handle_notes_title_changed(self, event)
        # --- Chat Sidebar Conversation Search ---
        elif input_id == "chat-conversation-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "chat-conversation-keyword-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "chat-conversation-tags-search-bar" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_conversation_search_bar_changed(self, event.value)
        elif input_id == "conv-char-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_search_input_changed(self, event)
        elif input_id == "conv-char-keyword-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_keyword_search_input_changed(self, event)
        elif input_id == "conv-char-tags-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_conversation_tags_search_input_changed(self, event)
        elif input_id == "ccp-prompt-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_prompt_search_input_changed(self, event)
        elif input_id == "ccp-worldbook-search-input" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_worldbook_search_input_changed(self, event)
        elif input_id == "chat-prompt-search-input" and current_active_tab == TAB_CHAT: # New condition
            if self._chat_sidebar_prompt_search_timer: # Use the new timer variable
                self._chat_sidebar_prompt_search_timer.stop()
            self._chat_sidebar_prompt_search_timer = self.set_timer(
                0.5,
                lambda: chat_handlers.handle_chat_sidebar_prompt_search_changed(self, event.value.strip())
            )
        elif input_id == "chat-character-search-input" and current_active_tab == TAB_CHAT:
            # No debouncer here, direct call as per existing handler
            await chat_handlers.handle_chat_character_search_input_changed(self, event)
        elif input_id == "chat-character-name-edit" and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_character_attribute_changed(self, event)
        elif input_id == "chat-template-search-input" and current_active_tab == TAB_CHAT:
            # No debouncer here, direct call for template search
            await chat_handlers.handle_chat_template_search_input_changed(self, event.value)
        elif input_id == "chat-llm-max-tokens" and current_active_tab == TAB_CHAT:
            # Update token counter when max tokens value changes
            self.call_after_refresh(self.update_token_count_display)
        elif input_id == "chat-custom-token-limit" and current_active_tab == TAB_CHAT:
            # Update token counter when custom token limit changes
            self.call_after_refresh(self.update_token_count_display)
        elif input_id == "chat-settings-search" and current_active_tab == TAB_CHAT:
            await self.handle_settings_search(event.value)
        # --- Chat Tab World Book Search Input ---
        elif input_id == "chat-worldbook-search-input" and current_active_tab == TAB_CHAT:
            await chat_events_worldbooks.handle_worldbook_search_input(self, event.value)
        # --- Chat Tab Dictionary Search Input ---
        elif input_id == "chat-dictionary-search-input" and current_active_tab == TAB_CHAT:
            await chat_events_dictionaries.handle_dictionary_search_input(self, event.value)
        # --- Chat Tab Media Search Input ---
        # elif input_id == "chat-media-search-input" and current_active_tab == TAB_CHAT:
        #     await handle_chat_media_search_input_changed(self, event.input)
        # --- Media Tab Search Inputs ---
        elif input_id and input_id.startswith("media-search-input-") and current_active_tab == TAB_MEDIA:
            await media_events.handle_media_search_input_changed(self, input_id, event.value)
        elif input_id and input_id.startswith("media-keyword-filter-") and current_active_tab == TAB_MEDIA:
            # Handle keyword filter changes with debouncing
            await media_events.handle_media_search_input_changed(self, input_id.replace("media-keyword-filter-", "media-search-input-"), event.value)
        # Add more specific input handlers if needed, e.g., for title inputs if they need live validation/reaction

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        list_view_id = event.list_view.id
        current_active_tab = self.current_tab
        item_details = f"Item prompt_id: {getattr(event.item, 'prompt_id', 'N/A')}, Item prompt_uuid: {getattr(event.item, 'prompt_uuid', 'N/A')}"
        self.loguru_logger.info(
            f"ListView.Selected: list_view_id='{list_view_id}', current_tab='{current_active_tab}', {item_details}"
        )

        if list_view_id and list_view_id.startswith("media-list-view-") and current_active_tab == TAB_MEDIA:
            self.loguru_logger.debug("Dispatching to media_events.handle_media_list_item_selected")
            await media_events.handle_media_list_item_selected(self, event)

        elif list_view_id == "notes-list-view" and current_active_tab == TAB_NOTES:
            self.loguru_logger.debug("Dispatching to notes_handlers.handle_notes_list_view_selected")
            await notes_handlers.handle_notes_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "ccp-prompts-listview" and current_active_tab == TAB_CCP:
            self.loguru_logger.debug("Dispatching to ccp_handlers.handle_ccp_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "chat-sidebar-prompts-listview" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Dispatching to chat_handlers.handle_chat_sidebar_prompts_list_view_selected")
            await ccp_handlers.handle_ccp_prompts_list_view_selected(self, list_view_id, event.item)

        elif list_view_id == "chat-media-search-results-listview" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Dispatching to chat_events_sidebar.handle_media_item_selected")
            await chat_events_sidebar.handle_media_item_selected(self, event.item)

        elif list_view_id == "chat-conversation-search-results-list" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Conversation selected in chat tab search results")
            # Store the selected item for the Load Selected button, but don't load immediately
            # This maintains the existing UX where users must click "Load Selected"
            
        elif list_view_id in ["chat-worldbook-available-listview", "chat-worldbook-active-listview"] and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug(f"World book selected in {list_view_id}")
            await chat_events_worldbooks.handle_worldbook_selection(self, list_view_id)
            
        elif list_view_id in ["chat-dictionary-available-listview", "chat-dictionary-active-listview"] and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug(f"Dictionary selected in {list_view_id}")
            await chat_events_dictionaries.handle_dictionary_selection(self, list_view_id)
            
        # Note: conv-char-search-results-list selections are handled by their respective "Load Selected" buttons.
        else:
            self.loguru_logger.warning(
            f"No specific handler for ListView.Selected from list_view_id='{list_view_id}' on tab='{current_active_tab}'")

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        checkbox_id = event.checkbox.id
        current_active_tab = self.current_tab

        if checkbox_id.startswith("chat-conversation-search-") and current_active_tab == TAB_CHAT:
            await chat_handlers.handle_chat_search_checkbox_changed(self, checkbox_id, event.value)
        elif checkbox_id.startswith("conv-char-search-") and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_search_checkbox_changed(self, checkbox_id, event.value)
        elif checkbox_id == "chat-show-attach-button-checkbox" and current_active_tab == TAB_CHAT:
            # Handle attach button visibility toggle
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat.images", "show_attach_button", event.value)
            
            # Update the UI if enhanced chat window is active
            use_enhanced_chat = get_cli_setting("chat_defaults", "use_enhanced_window", False)
            if use_enhanced_chat:
                try:
                    from .UI.Chat_Window_Enhanced import ChatWindowEnhanced
                    chat_window = self.query_one("#chat-window", ChatWindowEnhanced)
                    await chat_window.toggle_attach_button_visibility(event.value)
                except Exception as e:
                    loguru_logger.error(f"Error toggling attach button visibility: {e}")
        elif checkbox_id == "chat-show-dictation-button-checkbox" and current_active_tab == TAB_CHAT:
            # Handle dictation button visibility toggle
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat.voice", "show_mic_button", event.value)
            
            # Update the UI dynamically
            try:
                mic_button = self.query_one("#mic-button", Button)
                mic_button.display = event.value
                self.notify(f"Dictation button {'shown' if event.value else 'hidden'}", timeout=2)
            except QueryError:
                # If button doesn't exist, we'll need to refresh the chat window
                self.notify("Dictation button setting saved. Restart chat to apply changes.", timeout=3)
        elif checkbox_id == "chat-worldbook-enable-checkbox" and current_active_tab == TAB_CHAT:
            await chat_events_worldbooks.handle_worldbook_enable_checkbox(self, event.value)
        elif checkbox_id == "chat-dictionary-enable-checkbox" and current_active_tab == TAB_CHAT:
            await chat_events_dictionaries.handle_dictionary_enable_checkbox(self, event.value)
        elif checkbox_id == "chat-settings-mode-toggle" and current_active_tab == TAB_CHAT:
            # Handle settings mode toggle checkbox
            await self.handle_settings_mode_toggle_checkbox(event)
        # Add handlers for checkboxes in other tabs if any

    async def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handles changes in Switch widgets."""

        switch_id = event.switch.id
        current_active_tab = self.current_tab
        
        if switch_id == "notes-auto-save-toggle":
            await self.handle_notes_auto_save_toggle(event)


    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handles changes in Select widgets if specific actions are needed beyond watchers."""
        select_id = event.select.id
        current_active_tab = self.current_tab

        if select_id == "conv-char-character-select" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_character_select_changed(self, event.value)

        current_active_tab = self.current_tab

        if select_id == "conv-char-character-select" and current_active_tab == TAB_CCP:
            await ccp_handlers.handle_ccp_character_select_changed(self, event.value)
        elif select_id == "tldw-api-auth-method" and current_active_tab == TAB_INGEST:
            await ingest_events.handle_tldw_api_auth_method_changed(self, str(event.value))
        elif select_id == "tldw-api-media-type" and current_active_tab == TAB_INGEST:
            await ingest_events.handle_tldw_api_media_type_changed(self, str(event.value))
        elif select_id == "notes-sort-select" and current_active_tab == TAB_NOTES:
            await notes_handlers.handle_notes_sort_changed(self, event)
        elif select_id == "chat-rag-preset" and current_active_tab == TAB_CHAT:
            await self.handle_rag_preset_changed(event)
        elif select_id == "chat-rag-search-mode" and current_active_tab == TAB_CHAT:
            await self.handle_rag_pipeline_changed(event)
        elif select_id == "chat-rag-expansion-method" and current_active_tab == TAB_CHAT:
            await self.handle_query_expansion_method_changed(event)
        elif select_id == "chat-rag-expansion-provider" and current_active_tab == TAB_CHAT:
            # Update the reactive value to trigger the watcher
            self.rag_expansion_provider_value = event.value
        elif select_id == "chat-api-provider" and current_active_tab == TAB_CHAT:
            # Update token counter when provider changes
            try:
                from .Event_Handlers.Chat_Events.chat_token_events import update_chat_token_counter
                await update_chat_token_counter(self)
            except Exception as e:
                self.loguru_logger.debug(f"Could not update token counter on provider change: {e}")
        elif select_id == "chat-api-model" and current_active_tab == TAB_CHAT:
            # Update token counter when model changes
            try:
                from .Event_Handlers.Chat_Events.chat_token_events import update_chat_token_counter
                await update_chat_token_counter(self)
            except Exception as e:
                self.loguru_logger.debug(f"Could not update token counter on model change: {e}")
        elif select_id == "chat-conversation-search-character-filter-select" and current_active_tab == TAB_CHAT:
            self.loguru_logger.debug("Character filter changed in chat tab, triggering conversation search")
            await chat_handlers.perform_chat_conversation_search(self)
        elif select_id == "chat-rag-expansion-method" and current_active_tab == TAB_CHAT:
            # Handle query expansion method change - show/hide appropriate fields
            await self.handle_query_expansion_method_changed(event)

    ##################################################################
    # --- Event Handlers for Streaming and Worker State Changes ---
    ##################################################################
    @on(StreamingChunk)
    async def on_streaming_chunk(self, event: StreamingChunk) -> None:
        await handle_streaming_chunk(self, event)

    @on(StreamDone)
    async def on_stream_done(self, event: StreamDone) -> None:
        await handle_stream_done(self, event)
    
    @on(media_events.MediaMetadataUpdateEvent)
    async def on_media_metadata_update(self, event: media_events.MediaMetadataUpdateEvent) -> None:
        await media_events.handle_media_metadata_update(self, event)
    
    # Collections/Tags event handlers
    @on(Message)
    async def on_collections_tag_message(self, event: Message) -> None:
        """Handle Collections/Tag events."""
        from .Event_Handlers import collections_tag_events
        
        if event.__class__.__name__ == 'KeywordRenameEvent':
            await collections_tag_events.handle_keyword_rename(self, event)
        elif event.__class__.__name__ == 'KeywordMergeEvent':
            await collections_tag_events.handle_keyword_merge(self, event)
        elif event.__class__.__name__ == 'KeywordDeleteEvent':
            await collections_tag_events.handle_keyword_delete(self, event)
        elif event.__class__.__name__ == 'BatchAnalysisStartEvent':
            from .Event_Handlers import multi_item_review_events
            await multi_item_review_events.handle_batch_analysis_start(self, event)
        elif event.__class__.__name__ == 'TemplateDeleteConfirmationEvent':
            from .Widgets.confirmation_dialog import ConfirmationDialog
            from .Event_Handlers.template_events import TemplateDeleteConfirmationEvent
            
            if isinstance(event, TemplateDeleteConfirmationEvent):
                # Show confirmation dialog
                async def confirm_delete():
                    # Find the widget and call delete
                    try:
                        from .Widgets.chunking_templates_widget import ChunkingTemplatesWidget
                        # Find the templates widget in the current view
                        for widget in self.query(ChunkingTemplatesWidget):
                            widget.delete_template(event.template_id)
                    except Exception as e:
                        logger.error(f"Error deleting template: {e}")
                        self.notify(f"Error deleting template: {str(e)}", severity="error")
                
                dialog = ConfirmationDialog(
                    title="Delete Template",
                    message=f"Are you sure you want to delete the template '{event.template_name}'?\n\nThis action cannot be undone.",
                    confirm_label="Delete",
                    cancel_label="Cancel",
                    confirm_callback=confirm_delete
                )
                self.push_screen(dialog)
    
    @on(SplashScreen.Closed)
    async def on_splash_screen_closed(self, event: SplashScreen.Closed) -> None:
        """Handle splash screen closing."""
        self.splash_screen_active = False
        logger.debug("Splash screen closed, mounting main UI")
        
        # Remove the splash screen
        if self._splash_screen_widget:
            await self._splash_screen_widget.remove()
            self._splash_screen_widget = None
        
        # Check if main UI widgets already exist (avoid duplicate IDs)
        existing_ids = {widget.id for widget in self.screen._nodes if widget.id}
        
        # Create and mount the main UI components after splash screen is closed
        main_ui_widgets = self._create_main_ui_widgets()
        
        # Only mount widgets that don't already exist
        widgets_to_mount = []
        for widget in main_ui_widgets:
            if widget.id not in existing_ids:
                widgets_to_mount.append(widget)
            else:
                logger.debug(f"Skipping duplicate widget with ID: {widget.id}")
        
        if widgets_to_mount:
            await self.mount(*widgets_to_mount)
        
        # Now that the main UI is mounted, set up logging if it was deferred
        if not self._rich_log_handler:
            self.loguru_logger.debug("Setting up logging after splash screen closed")
            logging_start = time.perf_counter()
            self._setup_logging()
            if self._rich_log_handler:
                self.loguru_logger.debug("Starting RichLogHandler processor task...")
                self._rich_log_handler.start_processor(self)
            log_histogram("app_splash_deferred_logging_duration_seconds", time.perf_counter() - logging_start,
                         documentation="Time to set up logging after splash screen")
        
        # Now schedule post-mount setup and hide inactive windows
        self.call_after_refresh(self._post_mount_setup)
        self.call_after_refresh(self.hide_inactive_windows)
    

    @on(Checkbox.Changed, "#chat-strip-thinking-tags-checkbox")
    async def handle_strip_thinking_tags_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handles changes to the 'Strip Thinking Tags' checkbox."""
        new_value = event.value
        self.loguru_logger.info(f"'Strip Thinking Tags' checkbox changed to: {new_value}")

        if "chat_defaults" not in self.app_config:
            self.app_config["chat_defaults"] = {}
        self.app_config["chat_defaults"]["strip_thinking_tags"] = new_value

        # Persist the change
        try:
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat_defaults", "strip_thinking_tags", new_value)
            self.notify(f"Thinking tag stripping {'enabled' if new_value else 'disabled'}.", timeout=2)
        except Exception as e:
            self.loguru_logger.error(f"Failed to save 'strip_thinking_tags' setting: {e}", exc_info=True)
            self.notify("Error saving thinking tag setting.", severity="error", timeout=4)
    #####################################################################
    # --- End of Chat Event Handlers for Streaming & thinking tags ---
    #####################################################################

    async def handle_settings_mode_toggle_checkbox(self, event: Checkbox.Changed) -> None:
        """Handles the settings mode toggle checkbox between Basic and Advanced."""
        try:
            # Update reactive variable
            self.chat_settings_mode = "advanced" if event.value else "basic"
            
            # Update sidebar class for CSS styling
            sidebar = self.query_one("#chat-left-sidebar")
            if self.chat_settings_mode == "basic":
                sidebar.add_class("basic-mode")
                sidebar.remove_class("advanced-mode")
            else:
                sidebar.add_class("advanced-mode")
                sidebar.remove_class("basic-mode")
            
            # Notify user
            mode_name = "Advanced" if event.value else "Basic"
            self.notify(f"Switched to {mode_name} mode", timeout=2)
            
            # Save preference to config
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("chat_defaults", "advanced_mode", event.value)
            
        except Exception as e:
            loguru_logger.error(f"Error toggling settings mode: {e}", exc_info=True)
            self.notify("Error switching modes", severity="error", timeout=4)
    
    async def handle_settings_mode_toggle(self, event: Switch.Changed) -> None:
        """Handles the settings mode toggle between Basic and Advanced."""
        try:
            # Update reactive variable
            self.chat_settings_mode = "advanced" if event.value else "basic"
            
            # Update sidebar class for CSS styling
            sidebar = self.query_one("#chat-left-sidebar")
            if self.chat_settings_mode == "basic":
                sidebar.add_class("basic-mode")
                sidebar.remove_class("advanced-mode")
            else:
                sidebar.add_class("advanced-mode") 
                sidebar.remove_class("basic-mode")
            
            self.notify(f"Settings mode: {self.chat_settings_mode.title()}")
            self.loguru_logger.info(f"Switched to {self.chat_settings_mode} settings mode")
            
        except Exception as e:
            self.loguru_logger.error(f"Error toggling settings mode: {e}")
            self.notify("Error switching settings mode", severity="error")

    async def handle_notes_auto_save_toggle(self, event: Switch.Changed) -> None:
        """Handles the notes auto-save toggle."""
        try:
            # Update the reactive variable
            self.notes_auto_save_enabled = event.value
            
            # If auto-save is being disabled, cancel any pending timer
            if not event.value and self.notes_auto_save_timer is not None:
                self.notes_auto_save_timer.stop()
                self.notes_auto_save_timer = None
                self.loguru_logger.info("Auto-save timer cancelled")
            
            # Save the setting to config
            from .config import save_setting_to_cli_config
            save_setting_to_cli_config("notes", "auto_save_enabled", event.value)
            
            # Notify the user
            status = "enabled" if event.value else "disabled"
            self.notify(f"Notes auto-save {status}", timeout=2)
            self.loguru_logger.info(f"Notes auto-save {status}")
            
        except Exception as e:
            self.loguru_logger.error(f"Error toggling notes auto-save: {e}")
            self.notify("Error changing auto-save setting", severity="error")

    async def handle_rag_preset_changed(self, event: Select.Changed) -> None:
        """Handles RAG preset selection."""
        try:
            preset = event.value
            
            # Get RAG-related widgets
            rag_enable = self.query_one("#chat-rag-enable-checkbox", Checkbox)
            top_k = self.query_one("#chat-rag-top-k", Input)
            
            # Apply preset configurations
            if preset == "none":
                rag_enable.value = False
                self.notify("RAG disabled")
            elif preset == "light":
                rag_enable.value = True
                top_k.value = "3"
                self.notify("Light RAG: BM25 only, top 3 results")
            elif preset == "full":
                rag_enable.value = True
                top_k.value = "10"
                # Try to enable reranking if in advanced mode
                try:
                    rerank = self.query_one("#chat-rag-rerank-enable-checkbox", Checkbox)
                    rerank.value = True
                except QueryError:
                    pass  # Reranking is in advanced mode
                self.notify("Full RAG: Embeddings + reranking, top 10 results")
            elif preset == "custom":
                rag_enable.value = True
                self.notify("Custom RAG: Configure settings manually")
                
            self.loguru_logger.info(f"Applied RAG preset: {preset}")
            
        except Exception as e:
            self.loguru_logger.error(f"Error applying RAG preset: {e}")
            self.notify("Error applying RAG preset", severity="error")
    
    async def handle_rag_pipeline_changed(self, event: Select.Changed) -> None:
        """Handles RAG pipeline selection."""
        try:
            from .Widgets.settings_sidebar import get_pipeline_description
            
            pipeline_id = event.value
            
            # Update the description display
            description_widget = self.query_one("#chat-rag-pipeline-description", Static)
            description = get_pipeline_description(pipeline_id)
            description_widget.update(description)
            
            # If "none" is selected, just show manual config message
            if pipeline_id == "none":
                self.notify("Manual RAG configuration mode enabled")
            else:
                # Show what pipeline was selected
                pipeline_name = event.select.value_to_label(pipeline_id)
                self.notify(f"Selected pipeline: {pipeline_name}")
                
            self.loguru_logger.info(f"RAG pipeline changed to: {pipeline_id}")
            
        except Exception as e:
            self.loguru_logger.error(f"Error handling RAG pipeline change: {e}")
            self.notify("Error changing RAG pipeline", severity="error")
    
    async def handle_query_expansion_method_changed(self, event: Select.Changed) -> None:
        """Handles query expansion method selection - shows/hides appropriate fields."""
        try:
            method = event.value
            
            # Get the relevant widgets
            provider_label = self.query_one(".rag-expansion-provider-label", Static)
            provider_select = self.query_one("#chat-rag-expansion-provider", Select)
            llm_model_label = self.query_one(".rag-expansion-llm-label", Static)
            llm_model_select = self.query_one("#chat-rag-expansion-llm-model", Select)
            local_model_label = self.query_one(".rag-expansion-local-label", Static)
            local_model_input = self.query_one("#chat-rag-expansion-local-model", Input)
            
            # Show/hide based on method
            if method == "llm":
                # Show provider and model selection, hide local model
                provider_label.remove_class("hidden")
                provider_select.remove_class("hidden")
                llm_model_label.remove_class("hidden")
                llm_model_select.remove_class("hidden")
                local_model_label.add_class("hidden")
                local_model_input.add_class("hidden")
            elif method == "llamafile":
                # Show local model input, hide provider and model selection
                provider_label.add_class("hidden")
                provider_select.add_class("hidden")
                llm_model_label.add_class("hidden")
                llm_model_select.add_class("hidden")
                local_model_label.remove_class("hidden")
                local_model_input.remove_class("hidden")
            elif method == "keywords":
                # Hide all model fields
                provider_label.add_class("hidden")
                provider_select.add_class("hidden")
                llm_model_label.add_class("hidden")
                llm_model_select.add_class("hidden")
                local_model_label.add_class("hidden")
                local_model_input.add_class("hidden")
            
            self.loguru_logger.info(f"Query expansion method changed to: {method}")
            
        except Exception as e:
            self.loguru_logger.error(f"Error handling query expansion method change: {e}")
            self.notify("Error updating query expansion settings", severity="error")

    async def handle_settings_search(self, query: str) -> None:
        """Handles search in settings sidebar."""
        try:
            query = query.lower().strip()
            
            # Get all settings elements
            sidebar = self.query_one("#chat-left-sidebar")
            
            if not query:
                # Clear search - show all settings based on current mode
                for widget in sidebar.query(".sidebar-label, .section-header, .subsection-header"):
                    widget.remove_class("search-highlight")
                for collapsible in sidebar.query(Collapsible):
                    # Respect the original collapsed state
                    pass
                return
                
            # Search through all labels and highlight matches
            matches_found = 0
            
            for label in sidebar.query(".sidebar-label, .section-header, .subsection-header"):
                if isinstance(label, (Static, Label)):
                    label_text = str(label.renderable).lower()
                    if query in label_text:
                        label.add_class("search-highlight")
                        matches_found += 1
                        
                        # Expand parent collapsibles to show match
                        parent = label.parent
                        while parent and parent != sidebar:
                            if isinstance(parent, Collapsible):
                                parent.collapsed = False
                            parent = parent.parent
                    else:
                        label.remove_class("search-highlight")
                        
            if matches_found == 0:
                self.notify(f"No settings found for '{query}'", severity="warning")
            else:
                self.notify(f"Found {matches_found} settings matching '{query}'")
                
        except Exception as e:
            self.loguru_logger.error(f"Error in settings search: {e}")
            self.notify("Error searching settings", severity="error")


    #####################################################################
    # --- Event Handlers for Worker State Changes ---
    #####################################################################
    async def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """
        Handle worker state changes by delegating to the appropriate handler.
        
        This method has been refactored to use a handler registry pattern,
        significantly reducing complexity and improving maintainability.
        """
        worker_name = event.worker.name
        worker_group = event.worker.group
        
        # Log the state change
        self.loguru_logger.debug(
            f"on_worker_state_changed: Worker '{worker_name}' "
            f"(Group: {worker_group}, State: {event.state})"
        )
        
        # Delegate to the handler registry
        handled = await self.worker_handler_registry.handle_event(event)
        
        if not handled:
            # Log unhandled workers for debugging
            self.loguru_logger.warning(
                f"No handler found for worker '{worker_name}' (Group: {worker_group})"
            )
            
        # TODO: Fix this - new_user_prompt is not defined
        # try:
        #     self.query_one("#chat-prompt-user-display", TextArea).load_text(new_user_prompt or "")
        # except QueryError:
        #     self.loguru_logger.error("Chat sidebar user prompt display area (#chat-prompt-user-display) not found.")

    def _clear_chat_sidebar_prompt_display(self) -> None:
        """Clears the prompt display TextAreas in the chat sidebar."""
        UIHelpers.clear_chat_sidebar_prompt_display(self)

    def watch_chat_api_provider_value(self, new_value: Optional[str]) -> None:
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"Watcher: chat_api_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_model_select(TAB_CHAT, [])
            return
        models = self.providers_models.get(new_value, [])
        self._update_model_select(TAB_CHAT, models)

    def watch_ccp_api_provider_value(self, new_value: Optional[str]) -> None: # Renamed from watch_character_...
        if not hasattr(self, "app") or not self.app:  # Check if app is ready
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"Watcher: ccp_api_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_model_select(TAB_CCP, [])
            return
        models = self.providers_models.get(new_value, [])
        self._update_model_select(TAB_CCP, models)

    def watch_rag_expansion_provider_value(self, new_value: Optional[str]) -> None:
        """Watch for changes in RAG expansion provider selection."""
        if not hasattr(self, "app") or not self.app:
            return
        if not self._ui_ready:
            return
        self.loguru_logger.debug(f"Watcher: rag_expansion_provider_value changed to {new_value}")
        if new_value is None or new_value == Select.BLANK:
            self._update_rag_expansion_model_select([])
            return
        models = self.providers_models.get(new_value, [])
        self._update_rag_expansion_model_select(models)


    def _update_model_select(self, id_prefix: str, models: list[str]) -> None:
        if not self._ui_ready:  # Add guard
            return
        UIHelpers.update_model_select(self, id_prefix, models)

    def _update_rag_expansion_model_select(self, models: list[str]) -> None:
        """Update the RAG expansion model select options."""
        if not self._ui_ready:
            return
        UIHelpers.update_rag_expansion_model_select(self, models)

    def chat_wrapper(self, strip_thinking_tags: bool = True, **kwargs: Any) -> Any:
        """
        Delegates to the actual worker target function in worker_events.py.
        This method is called by app.run_worker.
        """
        # All necessary parameters (message, history, api_endpoint, model, etc.)
        # are passed via kwargs from the calling event handler (e.g., handle_chat_send_button_pressed).
        return worker_events.chat_wrapper_function(self, strip_thinking_tags=strip_thinking_tags, **kwargs) # Pass self as 'app_instance'

    def schedule_media_cleanup(self) -> None:
        """Schedule periodic media cleanup based on configuration."""
        try:
            # Get cleanup configuration
            cleanup_config = get_cli_setting("media_cleanup", "enabled", True)
            if not cleanup_config:
                self.loguru_logger.info("Media cleanup is disabled in configuration")
                return
                
            cleanup_interval_hours = get_cli_setting("media_cleanup", "cleanup_interval_hours", 24)
            cleanup_on_startup = get_cli_setting("media_cleanup", "cleanup_on_startup", True)
            
            # Run cleanup on startup if configured
            if cleanup_on_startup:
                self.loguru_logger.info("Running media cleanup on startup")
                self.call_later(self.perform_media_cleanup)
            
            # Schedule periodic cleanup
            cleanup_interval_seconds = cleanup_interval_hours * 3600
            self._media_cleanup_timer = self.set_interval(cleanup_interval_seconds, self.perform_media_cleanup)
            self.loguru_logger.info(f"Scheduled media cleanup every {cleanup_interval_hours} hours")
            
        except Exception as e:
            self.loguru_logger.error(f"Error scheduling media cleanup: {e}", exc_info=True)
    
    async def perform_media_cleanup(self) -> None:
        """Perform media cleanup based on configuration settings."""
        try:
            if not self.media_db:
                self.loguru_logger.warning("Media database not available for cleanup")
                return
                
            # Get cleanup configuration
            cleanup_days = get_cli_setting("media_cleanup", "cleanup_days", 30)
            max_items = get_cli_setting("media_cleanup", "max_items_per_cleanup", 100)
            notify_before = get_cli_setting("media_cleanup", "notify_before_cleanup", True)
            
            # Check for candidates first
            candidates = await asyncio.to_thread(self.media_db.get_deletion_candidates, cleanup_days)
            
            if not candidates:
                self.loguru_logger.info("No media items eligible for cleanup")
                return
                
            candidate_count = len(candidates)
            items_to_delete = min(candidate_count, max_items)
            
            # Notify user if configured
            if notify_before and candidate_count > 0:
                self.notify(
                    f"Found {candidate_count} media items eligible for permanent deletion "
                    f"(soft-deleted over {cleanup_days} days ago). "
                    f"Will delete up to {items_to_delete} items.",
                    title="Media Cleanup",
                    severity="information",
                    timeout=5
                )
            
            # Perform the cleanup
            deleted_count = await asyncio.to_thread(self.media_db.hard_delete_old_media, cleanup_days)
            
            if deleted_count > 0:
                self.loguru_logger.info(f"Media cleanup completed: {deleted_count} items permanently deleted")
                self.notify(
                    f"Media cleanup completed: {deleted_count} items permanently deleted",
                    severity="information",
                    timeout=3
                )
            
        except Exception as e:
            self.loguru_logger.error(f"Error during media cleanup: {e}", exc_info=True)
            self.notify(
                f"Error during media cleanup: {str(e)}",
                severity="error",
                timeout=5
            )
    
    def action_quit(self) -> None:
        """Handle application quit - save persistent caches before exiting."""
        loguru_logger.info("Application quit initiated")
        
        # Set flag to prevent new operations
        self._shutting_down = True
        
        # Force stop any playing audio and cleanup
        if hasattr(self, 'audio_player'):
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create cleanup tasks
                    async def cleanup_audio():
                        try:
                            await asyncio.wait_for(self.audio_player.stop(), timeout=0.5)
                        except asyncio.TimeoutError:
                            loguru_logger.warning("Audio stop timed out")
                        try:
                            await asyncio.wait_for(self.audio_player.cleanup(), timeout=0.5)
                        except asyncio.TimeoutError:
                            loguru_logger.warning("Audio cleanup timed out")
                    
                    # Schedule cleanup
                    asyncio.create_task(cleanup_audio())
                else:
                    # Synchronous cleanup if no event loop
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.audio_player.cleanup())
                    loop.close()
                loguru_logger.info("Audio player stopped and cleaned up")
            except Exception as e:
                loguru_logger.error(f"Error stopping audio during quit: {e}")
        
        # Force cleanup Higgs backends immediately
        if hasattr(self, '_stts_handler') and self._stts_handler:
            try:
                if hasattr(self._stts_handler, '_stts_service') and self._stts_handler._stts_service:
                    backend_manager = getattr(self._stts_handler._stts_service, 'backend_manager', None)
                    if backend_manager and hasattr(backend_manager, '_backends'):
                        for backend_id, backend in list(backend_manager._backends.items()):
                            if 'higgs' in backend_id.lower():
                                loguru_logger.info(f"Signaling Higgs backend shutdown: {backend_id}")
                                # Set shutdown event if available
                                if hasattr(backend, '_shutdown_event'):
                                    backend._shutdown_event.set()
            except Exception as e:
                loguru_logger.error(f"Error signaling Higgs shutdown: {e}")
        
        # Cancel media cleanup timer if it exists
        if hasattr(self, '_media_cleanup_timer') and self._media_cleanup_timer:
            self._media_cleanup_timer.stop()
        
        # Handle Notes auto-save cleanup
        if hasattr(self, 'notes_auto_save_timer') and self.notes_auto_save_timer is not None:
            self.notes_auto_save_timer.stop()
            self.notes_auto_save_timer = None
            loguru_logger.debug("Cancelled auto-save timer during app quit")
        
        # Perform final save if on Notes tab with unsaved changes (respect auto-save setting)
        if (self.current_tab == TAB_NOTES and 
            hasattr(self, 'notes_unsaved_changes') and 
            self.notes_unsaved_changes and 
            self.current_selected_note_id):
            # Check if we should save based on auto-save setting
            should_save = hasattr(self, 'notes_auto_save_enabled') and self.notes_auto_save_enabled
            if should_save:
                loguru_logger.debug("Performing final auto-save during app quit")
            else:
                loguru_logger.debug("Skipping final save during app quit (auto-save disabled)")
            
            if should_save:
                try:
                    # Import here to avoid circular imports
                    from tldw_chatbook.Event_Handlers.notes_events import save_current_note_handler
                    # Run synchronously since we're quitting
                    import asyncio
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(save_current_note_handler(self))
                    loop.close()
                except Exception as e:
                    loguru_logger.error(f"Error performing final save during quit: {e}")
        
        # Try to save caches but don't let it block quitting
        try:
            # Import with timeout protection
            import signal
            import threading
            
            def save_caches_with_timeout():
                # Note: The old cache service is deprecated
                # The simplified RAG service handles caching internally
                # and doesn't require explicit save on shutdown
                loguru_logger.debug("Cache saving skipped - handled by simplified RAG service")
            
            # Run cache saving in a separate thread with timeout
            save_thread = threading.Thread(target=save_caches_with_timeout)
            save_thread.daemon = True  # Don't let this thread prevent app exit
            save_thread.start()
            save_thread.join(timeout=2.0)  # Wait max 2 seconds
            
            if save_thread.is_alive():
                loguru_logger.warning("Cache save timed out - proceeding with quit")
        except Exception as e:
            loguru_logger.error(f"Error in quit handler: {e}")
        
        # Save encrypted config if encryption is enabled
        try:
            from tldw_chatbook.config import (
                load_cli_config_and_ensure_existence, 
                get_encryption_password,
                encrypt_api_keys_in_config,
                DEFAULT_CONFIG_PATH
            )
            from tldw_chatbook.Utils.atomic_file_ops import atomic_write_text
            import toml
            
            config_data = load_cli_config_and_ensure_existence()
            encryption_config = config_data.get("encryption", {})
            
            if encryption_config.get("enabled", False):
                password = get_encryption_password()
                if password:
                    loguru_logger.info("Encrypting configuration before exit...")
                    try:
                        # Encrypt the config
                        encrypted_config = encrypt_api_keys_in_config(config_data, password)
                        
                        # Save the encrypted config
                        config_text = toml.dumps(encrypted_config)
                        atomic_write_text(DEFAULT_CONFIG_PATH, config_text)
                        
                        loguru_logger.info("Configuration encrypted and saved successfully")
                    except Exception as e:
                        loguru_logger.error(f"Failed to encrypt config on exit: {e}")
                        # Continue with exit even if encryption fails
                else:
                    loguru_logger.warning("Encryption enabled but no password available - config not encrypted")
        except Exception as e:
            loguru_logger.error(f"Error during config encryption on exit: {e}")
        
        # Always call the parent quit method
        self.exit()

    ########################################################
    # --- End of Watchers and Helper Methods ---
    # ######################################################

# Initialize logging at the earliest possible point
def initialize_early_logging():
    """Initialize logging as early as possible to capture all logs from startup."""
    from .Logging_Config import configure_application_logging
    # Create a temporary app-like object with just enough attributes for configure_application_logging
    class EarlyLoggingApp:
        def __init__(self):
            self.app_config = load_settings()
            self._rich_log_handler = None

        def query_one(self, *args, **kwargs):
            # This will fail in configure_application_logging, but that's expected
            # for early logging - we just want to set up file and console logging
            raise QueryError("Early logging setup - UI not available yet")

    # Configure logging with our minimal app-like object
    early_app = EarlyLoggingApp()
    configure_application_logging(early_app)
    logging.info("Early logging initialization complete")
    loguru_logger.info("Early logging initialization complete (loguru)")
    return early_app

# --- Main execution block ---
if __name__ == "__main__":
    # Initialize logging first
    early_logging_app = initialize_early_logging()

    # Ensure config file exists (create default if missing)
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            logging.info(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default.")
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w", encoding='utf-8') as f:
                f.write(CONFIG_TOML_CONTENT)
    except Exception as e_cfg_main:
        logging.error(f"Could not ensure creation of default config file: {e_cfg_main}", exc_info=True)

    # --- Initialize Metrics Systems ---
    # Initialize Prometheus metrics server
    try:
        # Start Prometheus metrics server on port 8000 (or configure via env/config)
        metrics_port = int(os.environ.get('METRICS_PORT', '8000'))
        init_metrics_server(port=metrics_port)
        loguru_logger.info(f"Prometheus metrics server started on port {metrics_port}")
    except Exception as e:
        loguru_logger.warning(f"Failed to start Prometheus metrics server: {e}")
        # Continue without metrics server - metrics are still collected
    
    # Initialize OpenTelemetry metrics
    try:
        # Initialize OpenTelemetry for advanced metrics collection
        # This complements the existing Prometheus metrics
        init_otel_metrics()
        loguru_logger.info("OpenTelemetry metrics initialized successfully")
    except Exception as e:
        loguru_logger.warning(f"Failed to initialize OpenTelemetry metrics: {e}")
        # Continue without OpenTelemetry - the app still has Prometheus metrics

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji() # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}")
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    try:
        css_dir = Path(__file__).parent / "css"
        css_dir.mkdir(exist_ok=True)
        
        # Check if modular CSS needs to be built
        modular_css_path = css_dir / "tldw_cli_modular.tcss"
        build_script_path = css_dir / "build_css.py"
        
        # Check if any module is newer than the built file
        should_rebuild = False
        if not modular_css_path.exists():
            should_rebuild = True
            logging.info("Modular CSS file not found, will build it")
        elif build_script_path.exists():
            # Check if any module file is newer than the built file
            modular_mtime = modular_css_path.stat().st_mtime
            for subdir in ['core', 'layout', 'components', 'features', 'utilities']:
                subdir_path = css_dir / subdir
                if subdir_path.exists():
                    for css_file in subdir_path.glob('*.tcss'):
                        if css_file.stat().st_mtime > modular_mtime:
                            should_rebuild = True
                            logging.info(f"Module {css_file.name} is newer than built CSS, rebuilding")
                            break
                if should_rebuild:
                    break
        
        if should_rebuild and build_script_path.exists():
            logging.info("Building modular CSS...")
            import subprocess
            result = subprocess.run([sys.executable, str(build_script_path)], 
                                  cwd=str(css_dir), 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                logging.info("Successfully built modular CSS")
            else:
                logging.error(f"Failed to build modular CSS: {result.stderr}")
                # Fall back to legacy CSS if available
                from .Constants import css_content
                css_file_path = css_dir / "tldw_cli.tcss"
                if not css_file_path.exists():
                    with open(css_file_path, "w", encoding='utf-8') as f:
                        f.write(css_content)
                    logging.info(f"Created fallback CSS file: {css_file_path}")
        
    except Exception as e_css_main:
        logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # --- Check for encrypted config (config will be created if it doesn't exist) ---
    try:
        config_data = load_cli_config_and_ensure_existence()
        encryption_config = config_data.get("encryption", {})
        
        if encryption_config.get("enabled", False):
            loguru_logger.info("Config file encryption is enabled. Password required.")
            
            # Import password dialog dependencies here to avoid circular imports
            import asyncio
            from textual.app import App
            from tldw_chatbook.Widgets.password_dialog import PasswordDialog
            
            class PasswordPromptApp(App):
                """Minimal app to prompt for password."""
                def __init__(self):
                    super().__init__()
                    self.password = None
                
                async def on_mount(self) -> None:
                    """Show password dialog immediately on mount."""
                    password = await self.push_screen(
                        PasswordDialog(
                            mode="unlock",
                            title="Unlock Configuration",
                            message="Enter your master password to decrypt the configuration file.",
                            on_submit=lambda p: None,
                            on_cancel=lambda: None
                        ),
                        wait_for_dismiss=True
                    )
                    
                    if password:
                        # Verify password
                        from tldw_chatbook.Utils.config_encryption import config_encryption
                        password_verifier = encryption_config.get("password_verifier", "")
                        if password_verifier and config_encryption.verify_password(password, password_verifier):
                            self.password = password
                            self.exit()
                        else:
                            self.notify("Invalid password. Please try again.", severity="error")
                            # Re-show the dialog
                            await self.on_mount()
                    else:
                        # User cancelled
                        loguru_logger.error("Password required but not provided. Exiting.")
                        self.exit()
            
            # Run the password prompt app
            password_app = PasswordPromptApp()
            password_app.run()
            
            if password_app.password:
                # Set the password for the session
                set_encryption_password(password_app.password)
                loguru_logger.info("Configuration decrypted successfully.")
            else:
                # Exit if no password provided
                loguru_logger.error("Cannot proceed without decryption password.")
                sys.exit(1)
                
    except Exception as e:
        loguru_logger.error(f"Error checking config encryption: {e}")
        # Continue without encryption if there's an error

    # Create instance with early logging flag
    app_instance = TldwCli()
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
        # Force cleanup inline
        import threading
        import concurrent.futures
        for thread in threading.enumerate():
            if thread != threading.main_thread() and not thread.daemon:
                try:
                    thread.daemon = True
                except Exception:
                    pass
        try:
            concurrent.futures.thread._threads_queues.clear()
        except Exception:
            pass
    except Exception as e:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")

# Entry point for the tldw-chatbook command
def main_cli_runner():
    """Entry point for the tldw-chatbook command.

    This function is referenced in pyproject.toml as the entry point for the tldw-chatbook command.
    It initializes logging early and then runs the TldwCli app.
    """
    # Configure logging to suppress verbose debug messages early
    import logging
    import os
    import warnings
    
    # Suppress various verbose loggers
    logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    # Suppress torchaudio and FFmpeg warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", message=".*FFmpeg.*")
    
    # Set environment variable to suppress FFmpeg output
    os.environ["TORCHAUDIO_LOG_LEVEL"] = "ERROR"
    
    # Set up signal handlers for clean exit
    import signal
    import os
    import atexit
    
    def force_cleanup():
        """Force cleanup on exit"""
        import threading
        import concurrent.futures
        
        # Force kill any Higgs-related threads first
        for thread in threading.enumerate():
            thread_name = thread.name.lower()
            if any(name in thread_name for name in ['higgs', 'boson', 'serve_engine', 'audio']):
                loguru_logger.warning(f"Force killing thread: {thread.name}")
                try:
                    # Mark as daemon to not block exit
                    thread.daemon = True
                except Exception:
                    pass
        
        # Force daemon all threads
        for thread in threading.enumerate():
            if thread != threading.main_thread() and not thread.daemon:
                try:
                    thread.daemon = True
                except Exception:
                    pass
        
        # Clear thread pool queues
        try:
            concurrent.futures.thread._threads_queues.clear()
        except Exception:
            pass
        
        # Force clear any PyTorch CUDA resources
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
    
    # Register cleanup
    atexit.register(force_cleanup)
    
    def signal_handler(signum, frame):
        loguru_logger.info(f"Received signal {signum}, forcing clean exit")
        force_cleanup()
        os._exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize logging first
    early_logging_app = initialize_early_logging()

    # Ensure config file exists (create default if missing)
    try:
        if not DEFAULT_CONFIG_PATH.exists():
            logging.info(f"Config file not found at {DEFAULT_CONFIG_PATH}, creating default.")
            DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFAULT_CONFIG_PATH, "w", encoding='utf-8') as f:
                f.write(CONFIG_TOML_CONTENT)
    except Exception as e_cfg_main:
        logging.error(f"Could not ensure creation of default config file: {e_cfg_main}", exc_info=True)

    # --- Emoji Check ---
    emoji_is_supported = supports_emoji() # Call it once
    loguru_logger.info(f"Terminal emoji support detected: {emoji_is_supported}")
    loguru_logger.info(f"Using brain: {get_char(EMOJI_TITLE_BRAIN, FALLBACK_TITLE_BRAIN)}")
    loguru_logger.info("-" * 30)

    # --- CSS File Handling ---
    try:
        css_dir = Path(__file__).parent / "css"
        css_dir.mkdir(exist_ok=True)
        
        # Check if modular CSS needs to be built
        modular_css_path = css_dir / "tldw_cli_modular.tcss"
        build_script_path = css_dir / "build_css.py"
        
        # Check if any module is newer than the built file
        should_rebuild = False
        if not modular_css_path.exists():
            should_rebuild = True
            logging.info("Modular CSS file not found, will build it")
        elif build_script_path.exists():
            # Check if any module file is newer than the built file
            modular_mtime = modular_css_path.stat().st_mtime
            for subdir in ['core', 'layout', 'components', 'features', 'utilities']:
                subdir_path = css_dir / subdir
                if subdir_path.exists():
                    for css_file in subdir_path.glob('*.tcss'):
                        if css_file.stat().st_mtime > modular_mtime:
                            should_rebuild = True
                            logging.info(f"Module {css_file.name} is newer than built CSS, rebuilding")
                            break
                if should_rebuild:
                    break
        
        if should_rebuild and build_script_path.exists():
            logging.info("Building modular CSS...")
            import subprocess
            result = subprocess.run([sys.executable, str(build_script_path)], 
                                  cwd=str(css_dir), 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                logging.info("Successfully built modular CSS")
            else:
                logging.error(f"Failed to build modular CSS: {result.stderr}")
                # Fall back to legacy CSS if available
                from .Constants import css_content
                css_file_path = css_dir / "tldw_cli.tcss"
                if not css_file_path.exists():
                    with open(css_file_path, "w", encoding='utf-8') as f:
                        f.write(css_content)
                    logging.info(f"Created fallback CSS file: {css_file_path}")
        
    except Exception as e_css_main:
        logging.error(f"Error handling CSS file: {e_css_main}", exc_info=True)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="tldw chatbook - A Textual TUI for chatting with LLMs",
        prog="tldw-cli"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the application as a web server"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for web server (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for web server (default: 8000)"
    )
    parser.add_argument(
        "--web-title",
        type=str,
        help="Title for the web page"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for web server"
    )
    
    args = parser.parse_args()
    
    # If --serve flag is provided, run as web server
    if args.serve:
        # Check if web server dependencies are available
        from .Utils.optional_deps import DEPENDENCIES_AVAILABLE
        if not DEPENDENCIES_AVAILABLE.get('web', False):
            loguru_logger.error("\n" + "="*60)
            loguru_logger.error("Web server feature is not available!")
            loguru_logger.error("="*60)
            loguru_logger.error("\nThe required dependency 'textual-serve' is not installed.")
            loguru_logger.error("\nTo install it, run:")
            loguru_logger.error("  pip install tldw_chatbook[web]")
            loguru_logger.error("\nFor development installations:")
            loguru_logger.error("  pip install -e \".[web]\"")
            loguru_logger.error("\n" + "="*60 + "\n")
            return
        
        from .Web_Server.serve import run_web_server
        
        loguru_logger.info("Starting tldw_chatbook in web server mode")
        run_web_server(
            host=args.host,
            port=args.port,
            title=args.web_title,
            debug=args.debug
        )
        return  # Exit after web server stops
    
    # Otherwise, run as normal TUI app
    # Create instance with early logging flag
    app_instance = TldwCli()
    # Set the early logging flag so _setup_logging knows logging was already initialized
    app_instance._early_logging_initialized = True
    try:
        app_instance.run()
    except KeyboardInterrupt:
        loguru_logger.info("--- KeyboardInterrupt received ---")
        # Force cleanup inline
        import threading
        import concurrent.futures
        for thread in threading.enumerate():
            if thread != threading.main_thread() and not thread.daemon:
                try:
                    thread.daemon = True
                except Exception:
                    pass
        try:
            concurrent.futures.thread._threads_queues.clear()
        except Exception:
            pass
    except Exception as e:
        loguru_logger.exception("--- CRITICAL ERROR DURING app.run() ---")
        traceback.print_exc()  # Make sure traceback prints
    finally:
        # This might run even if app exits early internally in run()
        loguru_logger.info("--- FINALLY block after app.run() ---")

    loguru_logger.info("--- AFTER app.run() call (if not crashed hard) ---")

#
# End of app.py
#######################################################################################################################
