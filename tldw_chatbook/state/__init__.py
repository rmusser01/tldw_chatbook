"""Legacy caller-owned serialization compatibility containers.

The application keeps live state with its narrow runtime and screen owners.

Exports resolve lazily (PEP 562), as in ``Notifications/__init__.py``: only
``ui_state`` has a production consumer, and importing that leaf runs this
module -- which used to drag the four consumer-less siblings onto the boot
path with it (ADR-097 boot budget; tier-2 review S17 P3).
"""

from importlib import import_module

_EXPORTS = {
    "AppState": (".app_state", "AppState"),
    "NavigationState": (".navigation_state", "NavigationState"),
    "ChatState": (".chat_state", "ChatState"),
    "ChatSession": (".chat_state", "ChatSession"),
    "NotesState": (".notes_state", "NotesState"),
    "Note": (".notes_state", "Note"),
    "UIState": (".ui_state", "UIState"),
    "RuntimeSourceState": (
        "tldw_chatbook.runtime_policy.types",
        "RuntimeSourceState",
    ),
}

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


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
