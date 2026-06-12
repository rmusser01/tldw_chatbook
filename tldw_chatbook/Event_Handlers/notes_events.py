# notes_events.py
# Description:
#
# Imports
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING
import time
#
# 3rd-Party Imports
from loguru import logger
from textual.widgets import Input, ListView, TextArea, Label, Button, ListItem, Select, Static
from textual.css.query import QueryError  # For try-except
import yaml
#
# Local Imports
from ..DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, EnhancedFileSave as FileSave
from ..Third_Party.textual_fspicker import Filters
from ..config import load_cli_config_and_ensure_existence
#
if TYPE_CHECKING:
    from ..app import TldwCli
#
########################################################################################################################
#
# Functions:


########################################################################################################################
#
# Helper Functions (specific to Notes tab logic, moved from app.py)
#
########################################################################################################################


########################################################################################################################
#
# Helper Functions for Note Import
#
########################################################################################################################

def _parse_note_from_file_content(file_path: Path, file_content_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses note title and content from string content.
    Tries JSON (expects "title", "content" keys), then YAML (same keys),
    then plain text (first line as title, rest as content).
    For .txt and .md files, it directly uses the filename as title and full content.
    For other files, if JSON/YAML parsing fails or doesn't find a "title",
    it falls back to using the filename as title and the full file content.

    Returns:
        A tuple (title, content). Title might be None if unparsable and no filename fallback.
        Content will be the full string if unparsable.
    """
    logger_instance = logger  # Use global logger or pass app.loguru_logger
    filename_as_title = file_path.stem
    full_file_content = file_content_str.strip()

    if not full_file_content:  # Handle truly empty files after stripping
        logger_instance.debug(f"Note file '{file_path.name}' is empty. Using filename as title.")
        return filename_as_title, ""

    file_suffix = file_path.suffix.lower()

    # 1. Specific handling for .txt and .md files
    if file_suffix in ['.txt', '.md']:
        logger_instance.debug(f"Parsing note file '{file_path.name}' as TXT/MD. Using filename as title.")
        return filename_as_title, full_file_content

    # 1. Try JSON
    try:
        data = json.loads(file_content_str)
        if isinstance(data, dict):
            json_title = data.get("title")
            json_content = data.get("content")
            if json_title is not None:  # JSON has a "title" key
                logger_instance.debug(f"Parsed note file '{file_path.name}' as JSON.")
                # Use JSON content if present, otherwise use the full file content
                return json_title, json_content if json_content is not None else full_file_content
                # If JSON is valid dict but no "title", fall through to filename fallback logic
    except json.JSONDecodeError:
        logger_instance.debug(f"Note file '{file_path.name}' is not valid JSON. Trying YAML.")
    except Exception as e_json_other:
        logger_instance.warning(f"Unexpected error during JSON parsing of note '{file_path.name}': {e_json_other}")
        # Fall through

    # 2. Try YAML
    try:
        data = yaml.safe_load(file_content_str)
        if isinstance(data, dict):
            yaml_title = data.get("title")
            yaml_content = data.get("content")
            if yaml_title is not None:  # YAML has a "title" key
                logger_instance.debug(f"Parsed note file '{file_path.name}' as YAML.")
                return yaml_title, yaml_content if yaml_content is not None else full_file_content
            # If YAML is a valid dict but has no "title", fall through to filename fallback logic
    except yaml.YAMLError:
        logger_instance.debug(f"Note file '{file_path.name}' is not valid YAML. Using filename as title.")
    except Exception as e_yaml_other:
        logger_instance.warning(f"Unexpected error during YAML parsing of note '{file_path.name}': {e_yaml_other}")
    # Fall through

    # 3. Fallback for non-txt/md files if JSON/YAML parsing failed or didn't yield a title
    logger_instance.debug(
        f"Note file '{file_path.name}' (not TXT/MD) failed structured parsing or lacked 'title'. Using filename as title.")
    return filename_as_title, full_file_content


# --- Input/List View Changed Handlers for Notes Tab ---


# --- Export Handlers ---


# Button Handler Map will be defined at the end of the file

#
# --- Template Definitions ---

import json
from pathlib import Path

def load_note_templates():
    """Load note templates from JSON file or use defaults."""
    # Try to load from user's config directory first
    user_config_path = Path.home() / ".config" / "tldw_cli" / "note_templates.json"
    
    # Fallback to app's config directory
    app_config_path = Path(__file__).parent.parent / "Config_Files" / "note_templates.json"
    
    # Use hardcoded defaults as last resort
    default_templates = {
        "blank": {
            "title": "New Note",
            "content": "",
            "keywords": ""
        },
        "meeting": {
            "title": "Meeting Notes - {date}",
            "content": """## Meeting Notes

**Date:** {date}
**Time:** {time}
**Attendees:** 

### Agenda
- 

### Discussion Points
- 

### Action Items
- [ ] 
- [ ] 

### Next Steps
- 

### Notes
""",
            "keywords": "meeting, notes"
        }
    }
    
    # Try to load templates
    templates_data = None
    loaded_from = None
    
    # First try user config
    if user_config_path.exists():
        try:
            with open(user_config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                templates_data = data.get('templates', data)
                loaded_from = "user config"
        except Exception as e:
            logger.warning(f"Failed to load user templates from {user_config_path}: {e}")
    
    # Then try app config
    if templates_data is None and app_config_path.exists():
        try:
            with open(app_config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                templates_data = data.get('templates', data)
                loaded_from = "app config"
        except Exception as e:
            logger.warning(f"Failed to load app templates from {app_config_path}: {e}")
    
    # Use defaults if nothing loaded
    if templates_data is None:
        templates_data = default_templates
        loaded_from = "defaults"
    
    logger.info(f"Loaded {len(templates_data)} note templates from {loaded_from}")
    return templates_data

# Load templates on module import
NOTE_TEMPLATES = load_note_templates()

# --- New UX Enhancement Handlers ---


# --- Button Handler Map ---
# This must be defined after all handler functions

# End of notes_events.py
########################################################################################################################
