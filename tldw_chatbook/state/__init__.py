"""Legacy caller-owned serialization compatibility containers.

The application keeps live state with its narrow runtime and screen owners.
TASK-38613 (2026-11-02 audit): the four consumer-less compatibility
containers (``app_state``, ``chat_state``, ``navigation_state``,
``notes_state`` -- including ``AppState``'s stray plaintext
``encryption_password`` slot) were deleted; only ``ui_state``, the one leaf
with a production consumer (``UI/Screens/chat_screen.py``), remains.

Exports still resolve lazily (PEP 562), as in
``Notifications/__init__.py``, so importing the live leaf never drags other
modules onto the boot path (ADR-097 boot budget; tier-2 review S17 P3).
"""

from importlib import import_module

_EXPORTS = {
    "UIState": (".ui_state", "UIState"),
}

__all__ = [
    "UIState",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
