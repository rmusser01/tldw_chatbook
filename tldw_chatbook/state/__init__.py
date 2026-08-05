"""Legacy caller-owned serialization compatibility containers.

The application keeps live state with its narrow runtime and screen owners.
"""

from tldw_chatbook.runtime_policy.types import RuntimeSourceState

from .app_state import AppState
from .navigation_state import NavigationState
from .chat_state import ChatState, ChatSession
from .notes_state import NotesState, Note
from .ui_state import UIState

__all__ = [
    "AppState",
    "NavigationState",
    "ChatState",
    "ChatSession",
    "NotesState",
    "Note",
    "RuntimeSourceState",
    "UIState",
]
