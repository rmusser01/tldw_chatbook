"""``LibraryConversationsState`` -- the Conversations subsystem's own fields.

State PR of the Conversations extraction series (the decomposition
exemplar; ``backlog/docs/library-decomposition-recipe.md``,
``.superpowers/sdd/2026-09-01-library-decomposition-foundation`` task 6).
Every field here was moved verbatim out of ``LibraryScreen.__init__`` in
``tldw_chatbook/UI/Screens/library_screen.py`` -- same default, same type.
``library_screen.py`` keeps every original ``_library_conversation*``
attribute name alive as a generated getter/setter ``@property`` shim
pointing at ``self._conversations_state.<field>`` (see the sentinel-wrapped
block right after the ``LibraryScreen`` class body); a later controller PR
in this series moves the subsystem's methods here too.

Three fields (``reader_preferences``, ``reader_persistence_locks``,
``reader_layout``) keep their *original* ``__init__`` assignment line
completely untouched rather than being folded into this dataclass's own
constructor call, because their right-hand side is entangled with other
subsystems' initialization and cannot be relocated without touching code
outside Conversations' ownership:

- ``reader_preferences`` is one of eight targets unpacked from a single
  ``self._load_library_reader_preference_snapshot()`` call shared with
  Media, Collections, Notes, File Notes, Prompts and Skills.
- ``reader_persistence_locks`` depends on a local ``asyncio.Lock()``
  (``library_pane_persistence_lock``) also reused by five other
  subsystems' persistence-lock dicts.
- ``reader_layout`` is derived from ``reader_preferences`` after it settles
  from the shared snapshot above.

Those three lines keep assigning through the original attribute name
(e.g. ``self._library_conversation_reader_preferences = ...``); the
generated property shim silently routes the assignment into this
dataclass's field, so the observable end-of-``__init__`` state is
unchanged. Their dataclass defaults below are therefore momentary
placeholders, overwritten before anything else in ``__init__`` reads them.

``CONVERSATIONS_PLURAL_STATE_FIELDS`` (added task 8): the single
authoritative home for which field names use the plural
``_library_conversations_<name>`` shim prefix versus the singular
``_library_conversation_<name>`` prefix every other field uses.
``LibraryScreen``'s own generated shim block (task 6),
``LibraryConversationReaderController``'s shim block (task 7), and
``LibraryConversationsController``'s shim block (task 8) all import this
constant instead of each keeping its own literal copy -- task 7's fix
round 1 flagged the screen's and the reader controller's independent
copies of this same two-name set as a concrete, reviewer-identified drift
risk: a future field added to one copy and not the other fails silently,
as an ``AttributeError`` inside whichever moved body reaches for it first,
under the wrong prefix. One shared home closes that gap for good instead
of adding a third copy.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...Library.library_conversation_reader_state import (
    LIBRARY_CONVERSATION_PAGE_SIZE,
    ConversationReaderState,
)
from ...Library.row_selection import RowSelection
from ...Utils.adaptive_reader_state import (
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    resolve_adaptive_reader_layout,
)
from .screen_constants import LIBRARY_CONVERSATION_READER_PROFILE

#: See the module docstring's ``CONVERSATIONS_PLURAL_STATE_FIELDS`` note.
CONVERSATIONS_PLURAL_STATE_FIELDS: frozenset[str] = frozenset(
    {"row_selection", "select_mode"}
)


@dataclass
class LibraryConversationsState:
    """Every field the Conversations subsystem exclusively owns."""

    query: str = ""
    requested_page: int = 1
    requested_query: str = ""
    freshness: str = "uninitialized"
    stale_copy: str = ""
    selection_notice: str = ""
    focus_after_apply: str = ""
    page_records: tuple[Mapping[str, Any], ...] = ()
    page: int = 1
    page_size: int = LIBRARY_CONVERSATION_PAGE_SIZE
    total: int = 0
    total_known: bool = False
    has_more: bool = False
    page_loaded: bool = False
    loading: bool = False
    error: str = ""
    request_generation: int = 0
    select_mode: bool = False
    row_selection: RowSelection = field(
        default_factory=lambda: RowSelection("conversations")
    )
    reader_state: ConversationReaderState = field(
        default_factory=ConversationReaderState
    )
    reader_loaded_metadata: Mapping[str, Any] = field(default_factory=dict)
    reader_selected_metadata: Mapping[str, Any] = field(default_factory=dict)
    find_focus_intent: tuple[int, int, str] | None = None
    reader_mounted_authority: bool = False
    deleted_selection_id: str = ""

    # Placeholder defaults only -- see module docstring: the original
    # `__init__` line for each of these three still runs (through the
    # property shim) before anything reads it.
    reader_preferences: AdaptiveReaderLayoutPreferences = field(
        default_factory=AdaptiveReaderLayoutPreferences
    )
    reader_persistence_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    reader_layout: AdaptiveReaderEffectiveLayout = field(
        default_factory=lambda: resolve_adaptive_reader_layout(
            0,
            AdaptiveReaderLayoutPreferences(),
            LIBRARY_CONVERSATION_READER_PROFILE,
        )
    )
