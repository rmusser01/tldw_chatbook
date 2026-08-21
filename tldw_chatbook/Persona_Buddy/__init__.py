"""App-owned Persona Buddy state and preference contracts."""

from .controller import (
    BuddyDrainResult,
    PersonaBuddyController,
    PersonaBuddyLeaseToken,
    PersonaBuddySnapshot,
    PersonaBuddyVisualSnapshot,
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
    "parse_persona_buddy_preferences",
    "prepare_persona_buddy_frame",
    "persist_persona_buddy_preferences",
    "serialize_persona_buddy_preferences",
)
