"""``LibraryCollectionsState`` -- the Collections subsystem's own fields.

State PR of the Collections extraction series (wave-2 task 5,
``.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio``;
recipe: ``backlog/docs/library-decomposition-recipe.md``; the export
series -- ``library_export_state.py`` -- is the worked example this
mirrors, since it also needed no plural-prefix split). Every field here
was moved verbatim out of ``LibraryScreen.__init__`` in
``tldw_chatbook/UI/Screens/library_screen.py`` -- same default, same type.
``library_screen.py`` originally kept every original
``_library_collections_<field>`` attribute name alive as a generated
getter/setter ``@property`` shim pointing at
``self._collections_state.<field>`` (a sentinel-wrapped block right after
the ``LibraryScreen`` class body). The collections cleanup PR (task 7)
deleted that screen-side shim block entirely once the subsystem's methods
had all moved to ``LibraryCollectionsController`` (task 6) and the
screen's own remaining references were retargeted to call through that
controller instead. The controller that took over the subsystem's methods
carries its OWN generated shim block in its place -- reading/writing
through an injected ``collections_state_accessor`` rather than a direct
``self._collections_state`` attribute, since the controller does not hold
the state object itself. See the controller module's own shim-block
comment for why that block is permanent (not a cleanup-PR deletion
target, unlike the one this class's own state PR originally shared).

Every field uses the SAME ``_library_collections_`` prefix -- unlike
Conversations (whose subsystem name is singular, "conversation", so two
fields needed the plural ``_library_conversations_`` variant), Collections'
own subsystem name is already plural, and the ownership census found no
field needing a DIFFERENT prefix from the rest. So there is no
``COLLECTIONS_PLURAL_STATE_FIELDS`` constant here, matching Export's own
precedent (see the task-5 report's ownership table for the per-field
census that established this).

``_library_collections_capture_controller`` is NOT a field of this
dataclass -- it holds a live ``LibraryCollectionsCaptureController``
instance (wiring, not state; the ``_conversation_reader_controller``
precedent), so it stays a plain ``LibraryScreen`` attribute, constructed
at its original position, untouched by this move.

Three fields (``reader_preferences``, ``reader_persistence_locks``,
``reader_layout``) keep their *original* ``__init__`` assignment line
completely untouched rather than being folded into this dataclass's own
constructor call, because their right-hand side is entangled with other
subsystems' initialization and cannot be relocated without touching code
outside Collections' ownership -- exactly the same shape the conversations
exemplar's own state PR (``library_conversations_state.py``) documents for
its identically-named trio:

- ``reader_preferences`` is one of eight targets unpacked from a single
  ``self._load_library_reader_preference_snapshot()`` call shared with
  Media, Conversations, Notes, File Notes, Prompts and Skills.
- ``reader_persistence_locks`` depends on a local ``asyncio.Lock()``
  (``library_pane_persistence_lock``) also reused by five other
  subsystems' persistence-lock dicts.
- ``reader_layout`` is derived from ``reader_preferences`` after it settles
  from the shared snapshot above.

Because the shared tuple-unpack assignment for ``reader_preferences``
executes BEFORE any of Collections' other (non-entangled) fields are
assigned in the original ``__init__`` order, ``self._collections_state``
is constructed once, with no constructor arguments (every non-entangled
field's default is a static literal -- no computed default needed a
constructor argument, unlike Export's ``form``), at the same early point
``self._conversations_state`` is -- BEFORE the shared reader-preferences
tuple-unpack -- rather than at the position of the first non-entangled
field (which sits AFTER that unpack). This is a deliberate,
recipe-consistent deviation from the "construct at the position of the
first removed field" default: the underlying rule ("computed defaults
become constructor arguments so __init__ evaluation order is preserved")
is about behavioral equivalence, and constructing an all-static-default
dataclass earlier than its first non-entangled field has no observable
effect, whereas constructing it AFTER the entangled tuple-unpack would
raise ``AttributeError`` the first time that unpack's
``self._library_collections_reader_preferences`` target (routed through
the not-yet-installed shim into a not-yet-existing
``self._collections_state``) tried to assign. See the task-5 report for
the full reasoning and the exact line-order evidence.

All three entangled fields' dataclass defaults below are therefore
momentary placeholders, overwritten before anything else in ``__init__``
reads them -- identical in spirit to the conversations state object's own
three entangled-field placeholders.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ...Library.collections_capture_models import (
    CaptureCapabilities,
    CaptureHighlight,
    SavedCaptureSearch,
)
from ...Utils.adaptive_reader_state import (
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    resolve_adaptive_reader_layout,
)
from ...Widgets.Library import CollectionsReaderMode
from .screen_constants import LIBRARY_COLLECTIONS_READER_PROFILE


@dataclass
class LibraryCollectionsState:
    """Every field the Collections subsystem exclusively owns."""

    capture_capabilities: CaptureCapabilities | None = None
    saved_searches: tuple[SavedCaptureSearch, ...] = ()
    saved_searches_total: int = 0
    active_scope: str = "all"
    requested_page: int = 1
    reader_mode: CollectionsReaderMode = "read"
    highlights: tuple[CaptureHighlight, ...] = ()
    quick_capture_open: bool = False
    quick_capture_url: str = ""
    quick_capture_title: str = ""
    quick_capture_tags: str = ""
    quick_capture_note: str = ""
    save_outcome_unknown: bool = False
    confirming_save_retry: bool = False
    quick_capture_saving: bool = False
    filters_open: bool = False
    more_open: bool = False
    confirming_hard_delete: bool = False
    legacy_recovery_rows: int = 0
    legacy_recovery_open: bool = False
    legacy_recovery_lines: tuple[str, ...] = ()
    action_status: str = ""
    action_content: str = ""

    # Placeholder defaults only -- see module docstring. Post-cleanup
    # (wave-2 task 7 deleted the screen-side property shim), each of
    # these three is written directly onto this dataclass's own
    # instance in `LibraryScreen.__init__` -- `reader_preferences` via
    # the shared `_load_library_reader_preference_snapshot()` tuple-
    # unpack, `reader_layout` and `reader_persistence_locks` via later
    # direct `self._collections_state.<field> = ...` assignments -- not
    # routed through any shim, since none exists anymore.
    reader_preferences: AdaptiveReaderLayoutPreferences = field(
        default_factory=AdaptiveReaderLayoutPreferences
    )
    reader_persistence_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    reader_layout: AdaptiveReaderEffectiveLayout = field(
        default_factory=lambda: resolve_adaptive_reader_layout(
            0,
            AdaptiveReaderLayoutPreferences(),
            LIBRARY_COLLECTIONS_READER_PROFILE,
        )
    )
