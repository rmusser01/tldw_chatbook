# notes_events.py
# Description:
#
# Imports
import json
from pathlib import Path
from typing import TYPE_CHECKING

#
# 3rd-Party Imports
from loguru import logger

#
# Local Imports
#
if TYPE_CHECKING:
    pass
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




# --- Input/List View Changed Handlers for Notes Tab ---


# --- Export Handlers ---


# Button Handler Map will be defined at the end of the file

#
# --- Template Definitions ---

from pathlib import Path  # noqa: E402


def load_note_templates():
    """Load note templates from JSON file or use defaults."""
    from ..config import _get_effective_config_path

    # Try to load from user's config directory first
    user_config_path = _get_effective_config_path().parent / "note_templates.json"

    # Fallback to app's config directory
    app_config_path = (
        Path(__file__).parent.parent / "Config_Files" / "note_templates.json"
    )

    # Use hardcoded defaults as last resort
    default_templates = {
        "blank": {"title": "New Note", "content": "", "keywords": ""},
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
            "keywords": "meeting, notes",
        },
    }

    # Try to load templates
    templates_data = None
    loaded_from = None

    # First try user config
    if user_config_path.exists():
        try:
            from ..Notes.template_store import read_templates

            templates_data = read_templates()
            loaded_from = "user config"
        except Exception as e:
            logger.warning(
                f"Failed to load user templates from {user_config_path}: {e}"
            )

    # Then try app config
    if templates_data is None and app_config_path.exists():
        try:
            with open(app_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                templates_data = data.get("templates", data)
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
