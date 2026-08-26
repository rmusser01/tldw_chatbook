"""App-owned Persona Buddy state and preference contracts.

PEP-562 lazy facade (TASK-21103): this package init used to import
``controller``/``rendering`` eagerly, which dragged 93% of
``Persona_Visual`` (and with it ``PIL._imaging``, 1.28 s cold) onto the app
boot path through ``Chat/console_runtime.py``'s module-level import of the
stdlib-only ``console_adapter`` seam. Every public name still resolves at
``tldw_chatbook.Persona_Buddy.<name>``, but the defining submodule is only
executed on first attribute access. Guarded by
``Tests/Packaging/test_persona_buddy_import_closure.py``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .controller import (
        BuddyDrainResult,
        PersonaBuddyController,
        PersonaBuddyLeaseToken,
        PersonaBuddySnapshot,
        PersonaBuddyVisualSnapshot,
        load_local_persona_portrait,
    )
    from .preferences import (
        PERSONA_BUDDY_UNPOSITIONED_COORDINATE,
        PersonaBuddyGeometry,
        PersonaBuddyPreferences,
        PersonaBuddySelection,
        parse_persona_buddy_preferences,
        persist_persona_buddy_preferences,
        serialize_persona_buddy_preferences,
    )
    from .rendering import (
        PERSONA_BUDDY_FRAME_UNAVAILABLE,
        PersonaBuddyFrameError,
        PersonaBuddyPreparedFrame,
        prepare_persona_buddy_frame,
    )

_EXPORTS = {
    "BuddyDrainResult": "controller",
    "PersonaBuddyController": "controller",
    "PersonaBuddyLeaseToken": "controller",
    "PersonaBuddySnapshot": "controller",
    "PersonaBuddyVisualSnapshot": "controller",
    "load_local_persona_portrait": "controller",
    "PERSONA_BUDDY_UNPOSITIONED_COORDINATE": "preferences",
    "PersonaBuddyGeometry": "preferences",
    "PersonaBuddyPreferences": "preferences",
    "PersonaBuddySelection": "preferences",
    "parse_persona_buddy_preferences": "preferences",
    "persist_persona_buddy_preferences": "preferences",
    "serialize_persona_buddy_preferences": "preferences",
    "PERSONA_BUDDY_FRAME_UNAVAILABLE": "rendering",
    "PersonaBuddyFrameError": "rendering",
    "PersonaBuddyPreparedFrame": "rendering",
    "prepare_persona_buddy_frame": "rendering",
}

__all__ = (
    "PERSONA_BUDDY_UNPOSITIONED_COORDINATE",
    "PERSONA_BUDDY_FRAME_UNAVAILABLE",
    "BuddyDrainResult",
    "PersonaBuddyController",
    "PersonaBuddyGeometry",
    "PersonaBuddyFrameError",
    "PersonaBuddyLeaseToken",
    "PersonaBuddyPreferences",
    "PersonaBuddyPreparedFrame",
    "PersonaBuddySelection",
    "PersonaBuddySnapshot",
    "PersonaBuddyVisualSnapshot",
    "load_local_persona_portrait",
    "parse_persona_buddy_preferences",
    "prepare_persona_buddy_frame",
    "persist_persona_buddy_preferences",
    "serialize_persona_buddy_preferences",
)


def __getattr__(name: str) -> object:
    """Resolve a public export by importing its defining submodule on demand.

    Args:
        name: The requested package attribute.

    Returns:
        The submodule attribute the facade re-exports.

    Raises:
        AttributeError: If ``name`` is not a public Persona_Buddy export.
    """
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the package's public exports for introspection."""
    return sorted(set(globals()) | set(__all__))
