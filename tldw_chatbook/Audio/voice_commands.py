# voice_commands.py
"""
Expanded voice command system for dictation with customizable commands.
Supports app navigation, text manipulation, and custom user-defined commands.
"""

import re
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..config import get_cli_setting, save_setting_to_cli_config


class CommandType(Enum):
    """Types of voice commands."""
    TEXT_MANIPULATION = "text_manipulation"
    APP_NAVIGATION = "app_navigation"
    PUNCTUATION = "punctuation"
    FORMATTING = "formatting"
    CUSTOM = "custom"


@dataclass
class VoiceCommand:
    """Definition of a voice command."""
    phrase: str  # The phrase to detect
    action: str  # The action identifier
    command_type: CommandType
    parameters: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    enabled: bool = True
    # For custom commands
    callback: Optional[Callable] = None
    regex_pattern: Optional[str] = None  # For more complex matching


class VoiceCommandProcessor:
    """
    Processes voice commands with support for built-in and custom commands.
    """
    
    # Built-in command definitions
    BUILTIN_COMMANDS = [
        # Text manipulation
        VoiceCommand(
            "new paragraph",
            "insert_paragraph",
            CommandType.TEXT_MANIPULATION,
            description="Insert a new paragraph"
        ),
        VoiceCommand(
            "new line",
            "insert_line",
            CommandType.TEXT_MANIPULATION,
            description="Insert a line break"
        ),
        VoiceCommand(
            "delete last word",
            "delete_word",
            CommandType.TEXT_MANIPULATION,
            description="Delete the last word"
        ),
        VoiceCommand(
            "delete last sentence",
            "delete_sentence", 
            CommandType.TEXT_MANIPULATION,
            description="Delete the last sentence"
        ),
        VoiceCommand(
            "clear all",
            "clear_all",
            CommandType.TEXT_MANIPULATION,
            description="Clear all text"
        ),
        VoiceCommand(
            "undo that",
            "undo",
            CommandType.TEXT_MANIPULATION,
            description="Undo last action"
        ),
        
        # Punctuation
        VoiceCommand(
            "comma",
            "insert_comma",
            CommandType.PUNCTUATION,
            parameters={"text": ","}
        ),
        VoiceCommand(
            "period",
            "insert_period",
            CommandType.PUNCTUATION,
            parameters={"text": "."}
        ),
        VoiceCommand(
            "question mark",
            "insert_question",
            CommandType.PUNCTUATION,
            parameters={"text": "?"}
        ),
        VoiceCommand(
            "exclamation mark",
            "insert_exclamation",
            CommandType.PUNCTUATION,
            parameters={"text": "!"}
        ),
        VoiceCommand(
            "semicolon",
            "insert_semicolon",
            CommandType.PUNCTUATION,
            parameters={"text": ";"}
        ),
        VoiceCommand(
            "colon",
            "insert_colon",
            CommandType.PUNCTUATION,
            parameters={"text": ":"}
        ),
        VoiceCommand(
            "open quote",
            "insert_open_quote",
            CommandType.PUNCTUATION,
            parameters={"text": '"'}
        ),
        VoiceCommand(
            "close quote",
            "insert_close_quote",
            CommandType.PUNCTUATION,
            parameters={"text": '"'}
        ),
        
        # Formatting
        VoiceCommand(
            "make bold",
            "format_bold",
            CommandType.FORMATTING,
            description="Make selected text bold"
        ),
        VoiceCommand(
            "make italic",
            "format_italic",
            CommandType.FORMATTING,
            description="Make selected text italic"
        ),
        VoiceCommand(
            "capitalize that",
            "capitalize",
            CommandType.FORMATTING,
            description="Capitalize selected text"
        ),
        VoiceCommand(
            "uppercase that",
            "uppercase",
            CommandType.FORMATTING,
            description="Convert to uppercase"
        ),
        VoiceCommand(
            "lowercase that",
            "lowercase",
            CommandType.FORMATTING,
            description="Convert to lowercase"
        ),
        
        # App navigation
        VoiceCommand(
            "stop dictation",
            "stop_dictation",
            CommandType.APP_NAVIGATION,
            description="Stop dictation"
        ),
        VoiceCommand(
            "pause dictation",
            "pause_dictation",
            CommandType.APP_NAVIGATION,
            description="Pause dictation"
        ),
        VoiceCommand(
            "switch to chat",
            "switch_to_chat",
            CommandType.APP_NAVIGATION,
            description="Switch to chat tab"
        ),
        VoiceCommand(
            "switch to notes",
            "switch_to_notes",
            CommandType.APP_NAVIGATION,
            description="Switch to notes tab"
        ),
        VoiceCommand(
            "save this",
            "save_current",
            CommandType.APP_NAVIGATION,
            description="Save current content"
        ),
        VoiceCommand(
            "show help",
            "show_help",
            CommandType.APP_NAVIGATION,
            description="Show help"
        ),
    ]
    
    # Regex patterns for more complex commands
    REGEX_COMMANDS = [
        # "Insert [text] at the beginning"
        (
            r"insert (.+) at the beginning",
            "insert_at_start",
            CommandType.TEXT_MANIPULATION
        ),
        # "Replace [old] with [new]"
        (
            r"replace (.+) with (.+)",
            "replace_text",
            CommandType.TEXT_MANIPULATION
        ),
        # "Go to line [number]"
        (
            r"go to line (\d+)",
            "goto_line",
            CommandType.APP_NAVIGATION
        ),
        # "Select from [start] to [end]"
        (
            r"select from (.+) to (.+)",
            "select_range",
            CommandType.TEXT_MANIPULATION
        ),
    ]
    
    def __init__(self):
        """Initialize command processor."""
        # Load built-in commands
        self.commands: Dict[str, VoiceCommand] = {}
        for cmd in self.BUILTIN_COMMANDS:
            self.commands[cmd.phrase.lower()] = cmd
        
        # Load custom commands from config
        self._load_custom_commands()
        
        # Compile regex patterns
        self.regex_patterns = []
        for pattern, action, cmd_type in self.REGEX_COMMANDS:
            self.regex_patterns.append((
                re.compile(pattern, re.IGNORECASE),
                action,
                cmd_type
            ))
    
    def _load_custom_commands(self):
        """Load user-defined custom commands from configuration."""
        custom_commands = get_cli_setting('dictation.custom_commands', [])
        
        for cmd_data in custom_commands:
            try:
                command = VoiceCommand(
                    phrase=cmd_data['phrase'],
                    action=cmd_data['action'],
                    command_type=CommandType.CUSTOM,
                    parameters=cmd_data.get('parameters'),
                    description=cmd_data.get('description'),
                    enabled=cmd_data.get('enabled', True)
                )
                
                if command.enabled:
                    self.commands[command.phrase.lower()] = command
                    
            except Exception as e:
                logger.error(f"Failed to load custom command: {e}")
    
    def save_custom_command(self, command: VoiceCommand):
        """Save a custom command to configuration."""
        custom_commands = get_cli_setting('dictation.custom_commands', [])
        
        # Convert command to dict
        cmd_data = {
            'phrase': command.phrase,
            'action': command.action,
            'parameters': command.parameters,
            'description': command.description,
            'enabled': command.enabled
        }
        
        # Update or add
        found = False
        for i, existing in enumerate(custom_commands):
            if existing['phrase'] == command.phrase:
                custom_commands[i] = cmd_data
                found = True
                break
        
        if not found:
            custom_commands.append(cmd_data)
        
        # Save
        save_setting_to_cli_config('dictation', 'custom_commands', custom_commands)
        
        # Update runtime
        if command.enabled:
            self.commands[command.phrase.lower()] = command
        elif command.phrase.lower() in self.commands:
            del self.commands[command.phrase.lower()]
    
    def delete_custom_command(self, phrase: str):
        """Delete a custom command."""
        custom_commands = get_cli_setting('dictation.custom_commands', [])
        custom_commands = [
            cmd for cmd in custom_commands 
            if cmd['phrase'].lower() != phrase.lower()
        ]
        save_setting_to_cli_config('dictation', 'custom_commands', custom_commands)
        
        # Remove from runtime
        if phrase.lower() in self.commands:
            cmd = self.commands[phrase.lower()]
            if cmd.command_type == CommandType.CUSTOM:
                del self.commands[phrase.lower()]
    
    def process_text(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Process text for voice commands.
        
        Returns:
            Tuple of (action, parameters) if command found, None otherwise
        """
        text_lower = text.lower().strip()
        
        # Check exact matches first
        if text_lower in self.commands:
            command = self.commands[text_lower]
            if command.enabled:
                return (command.action, command.parameters or {})
        
        # Check if text ends with any command
        for phrase, command in self.commands.items():
            if command.enabled and text_lower.endswith(phrase):
                # Extract the text before the command
                prefix = text[:-(len(phrase))].strip()
                params = command.parameters.copy() if command.parameters else {}
                if prefix:
                    params['prefix_text'] = prefix
                return (command.action, params)
        
        # Check regex patterns
        for pattern, action, cmd_type in self.regex_patterns:
            match = pattern.search(text)
            if match:
                params = {'groups': match.groups()}
                return (action, params)
        
        return None
    
    def get_command_list(self, command_type: Optional[CommandType] = None) -> List[VoiceCommand]:
        """Get list of available commands, optionally filtered by type."""
        commands = list(self.commands.values())
        
        if command_type:
            commands = [cmd for cmd in commands if cmd.command_type == command_type]
        
        return sorted(commands, key=lambda c: c.phrase)
    
    def get_command_help(self) -> Dict[str, List[str]]:
        """Get help text organized by command type."""
        help_dict = {}
        
        for cmd_type in CommandType:
            commands = self.get_command_list(cmd_type)
            if commands:
                help_dict[cmd_type.value] = [
                    f"'{cmd.phrase}'" + (f" - {cmd.description}" if cmd.description else "")
                    for cmd in commands
                ]
        
        return help_dict
    
    def is_command_enabled(self, phrase: str) -> bool:
        """Check if a command is enabled."""
        cmd = self.commands.get(phrase.lower())
        return cmd.enabled if cmd else False
    
    def toggle_command(self, phrase: str, enabled: bool):
        """Enable or disable a command."""
        if phrase.lower() in self.commands:
            self.commands[phrase.lower()].enabled = enabled
            
            # Save if it's a custom command
            if self.commands[phrase.lower()].command_type == CommandType.CUSTOM:
                self.save_custom_command(self.commands[phrase.lower()])


# Global instance
_command_processor = None

def get_command_processor() -> VoiceCommandProcessor:
    """Get the global command processor instance."""
    global _command_processor
    if _command_processor is None:
        _command_processor = VoiceCommandProcessor()
    return _command_processor