"""``LibrarySkillsState`` -- the Skills subsystem's own fields.

State PR of the Skills extraction series (wave-4 task 1, recipe:
``backlog/docs/library-decomposition-recipe.md``; the search+RAG series --
``library_rag_search_state.py`` -- is the worked example this mirrors most
closely, since it also needed a mixed-prefix shim mapping). Every field here
was moved verbatim out of ``LibraryScreen.__init__`` in
``tldw_chatbook/UI/Screens/library_screen.py`` -- same default, same type.
``library_screen.py`` originally kept every original ``_library_skill_
<field>``/``_library_skills_<field>``/``_selected_skill_name`` attribute
name alive as a generated getter/setter ``@property`` shim pointing at
``self._skills_state.<field>`` (a sentinel-wrapped block right after the
``LibraryScreen`` class body). The skills cleanup PR (task 3) deleted that
screen-side shim block entirely once the subsystem's methods had all moved
to ``LibrarySkillsController`` (task 2) and the screen's own remaining
references were retargeted to call through that controller instead. The
controller that took over the subsystem's methods carries its OWN
generated shim block in its place -- reading/writing through an injected
``skills_state_accessor`` rather than a direct ``self._skills_state``
attribute, since the controller does not hold the state object itself. See
the controller module's own shim-block comment for why that block is
permanent (not a cleanup-PR deletion target, unlike the one this class's
own state PR originally shared) -- the same export/collections/search+RAG
precedent.

Two existing, already-extracted Skills modules were NOT touched by either
move: ``library_skill_import_controller.py`` (``LibrarySkillImportCoordinator``)
and ``library_skills_browse_controller.py`` (``LibrarySkillsBrowseController``).
Screen methods that merely delegated to either were exclusion candidates
for the controller-move task (task 2), not this state task's own concern.

Three prefixes, not two -- a genuine deviation from the plan's own "two-prefix
mapping" framing, found by running the recipe's own field-ownership script
(``backlog/docs/library-decomposition-recipe.md`` §2) rather than assuming
the plan's shorthand was exhaustive:

- ``_library_skill_`` (singular) is the DEFAULT prefix -- 26 of the 36
  fields (e.g. ``dirty``, ``status``, ``editor_state``).
- ``_library_skills_`` (plural) covers 9 fields that describe the skills
  LIST/browse surface as a whole rather than one skill's own editor state:
  ``sort``, ``filter``, ``filter_cursor_context``, ``view``,
  ``sort_choices_visible``, ``trust_posture``, plus the reader-layout trio
  below. ``SKILLS_PLURAL_STATE_FIELDS`` is the single authoritative home
  for this set (mirroring the conversations exemplar's own
  ``CONVERSATIONS_PLURAL_STATE_FIELDS``/the search+RAG series' own
  ``SEARCH_PREFIXED_STATE_FIELDS`` -- one shared copy, not independently
  drifting duplicates in the screen's own shim generator and this module).
- ``_`` (bare underscore, no "skill"/"skills" word at all) covers exactly
  ONE field: ``selected_skill_name`` (the original attribute is
  ``_selected_skill_name`` -- found only because the ownership census was
  run as a substring match on "skill" rather than a ``startswith`` filter
  on the two `_library_skill(s)_` prefixes; the conversations exemplar's
  own recipe §11 "startswith enumeration trap" lesson, reproduced here on
  a THIRD prefix shape instead of a missed field within a known prefix).

``skill_state_shim_attr()`` below is the single function both the screen's
shim-generator loop and this subsystem's wiring test call to resolve a
field name to its original attribute name, rather than each independently
recomputing the three-way branch (the drift risk the frozenset-copy lesson
above already warns about, applied to the whole prefix decision, not just
the plural set).

Two fields are WIRING, not state, and are NOT part of this dataclass at
all -- exactly the ``_conversation_reader_controller``/
``_library_collections_capture_controller`` "capture-controller" precedent:

- ``_library_skill_import_coordinator`` holds the live
  ``LibrarySkillImportCoordinator`` singleton (shared across screen
  instances via ``ensure_library_skill_import_coordinator``); constructed
  at its original position, untouched by this move.
- ``_library_skills_browse_controller`` holds the live
  ``LibrarySkillsBrowseController`` instance; also constructed at its
  original position, untouched.

Three fields (``reader_preferences``, ``reader_persistence_locks``,
``reader_layout``) keep their *original* ``__init__`` assignment line
completely untouched, exactly the same shape the collections/conversations/
search+RAG series' own identically-named trios document:

- ``reader_preferences`` is one of eight targets unpacked from a single
  ``self._load_library_reader_preference_snapshot()`` call shared with
  Media, Conversations, Notes, File Notes, Collections and Prompts.
- ``reader_persistence_locks`` depends on a local ``asyncio.Lock()``
  (``library_pane_persistence_lock``) also reused by five other
  subsystems' persistence-lock dicts.
- ``reader_layout`` is derived from ``reader_preferences`` (via
  ``resolve_adaptive_reader_layout``) after it settles from the shared
  snapshot above.

**A new wrinkle this series adds to that precedent**: two MORE fields
(``editor_mode``, ``reader_mode``) also keep their original ``__init__``
lines untouched, even though neither is entangled with another
subsystem's own initialization code the way the trio is. Both call a
same-subsystem pure function (``coerce_skill_editor_mode``/
``coerce_skill_reader_mode``, imported from the pre-existing
``tldw_chatbook.Library.library_skills_state`` module -- a DIFFERENT
module from this one; see the note at the bottom of this docstring) with
real runtime config data (``editor_mode``) or a literal (``reader_mode``).
The reason both must still keep their original line is positional, not
entanglement: this dataclass's own instance (``self._skills_state``) has
to be constructed BEFORE the reader-preferences tuple-unpack (line 2392 at
this task's measurement) for the SAME reason the collections series'
own state PR documents -- the trio's original lines would otherwise route
through a not-yet-installed property into a not-yet-existing state object.
That forces construction to happen at line ~2163 (right after
``self._collections_state = LibraryCollectionsState()``), which is BEFORE
``editor_mode``'s (line 2913) and ``reader_mode``'s (line 2995) own
original lines. A field positioned after the forced-early construction
point cannot be "passed as a constructor argument" (the recipe's usual
computed-default rule) because construction has already happened by the
time its line runs -- so the ONLY behaviorally-transparent option is the
same one the trio already uses: leave the original line exactly as it
was, let the newly-installed property shim silently route the assignment
into the state object, same as the trio. Generalizes the trio's own
mechanism (previously described as being ABOUT entanglement) to any field
whose original position simply falls after the forced construction point,
entangled or not. Every other field's original line falls BEFORE that
forced construction point in the OLD ownership sense only for
``_library_skill_import_coordinator`` (WIRING, stays) and
``_library_skill_choice_presented_generation`` (static ``-1``, folded
into this dataclass's own default, its original line deleted) --
everything else here sits after, hence needs one of the two treatments
above (static-default fold or original-line-untouched).

All five entangled/computed fields' dataclass defaults below are
therefore momentary placeholders, overwritten before anything else in
``__init__`` reads them -- identical in spirit to the collections/
conversations/search+RAG state objects' own entangled-field placeholders.

Note on the two ``library_skills_state`` modules: this file
(``tldw_chatbook/UI/Library_Modules/library_skills_state.py``) is the
screen's OWN ``__init__``-field state object, following the exact
directory precedent every other subsystem uses (``UI/Library_Modules/``
pairs with an unrelated, pre-existing ``tldw_chatbook/Library/`` module of
the SAME basename for six other subsystems already --
``library_conversations_state.py``, ``library_export_state.py``, etc. --
so this is not a new pattern). The OTHER, pre-existing
``tldw_chatbook.Library.library_skills_state`` module (a pure-function/
pure-dataclass library with no Textual/DB/network imports, covered by its
own ``Tests/Library/test_library_skills_state.py``) supplies
``SkillEditorState``, ``SkillEditorMode``, ``SkillReaderMode``,
``coerce_skill_editor_mode`` and ``coerce_skill_reader_mode`` -- imported
here by fully-qualified path, so there is no namespace collision despite
the shared basename.
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...Library.library_skills_state import (
    SkillEditorMode,
    SkillEditorState,
    SkillReaderMode,
)
from ...Utils.adaptive_reader_state import (
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    resolve_adaptive_reader_layout,
)
from .screen_constants import LIBRARY_SKILLS_READER_PROFILE

#: See the module docstring's note on this constant: the single
#: authoritative home for which field names use the plural
#: ``_library_skills_`` shim prefix instead of the cluster's default
#: singular ``_library_skill_`` prefix.
SKILLS_PLURAL_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "sort",
        "filter",
        "filter_cursor_context",
        "view",
        "sort_choices_visible",
        "trust_posture",
        "reader_preferences",
        "reader_layout",
        "reader_persistence_locks",
    }
)

#: The single field using the bare ``_`` prefix (no "skill"/"skills" word
#: at all in the original attribute name).
SKILL_UNPREFIXED_STATE_FIELDS: frozenset[str] = frozenset({"selected_skill_name"})


def skill_state_shim_attr(field_name: str) -> str:
    """The original ``LibraryScreen`` attribute name for a state field.

    Single-source resolution of the three-way prefix mapping documented in
    this module's own docstring -- used by BOTH the screen's generated
    shim loop and this subsystem's wiring test, so the mapping cannot
    drift into two independently-typed copies.
    """
    if field_name in SKILL_UNPREFIXED_STATE_FIELDS:
        return "_" + field_name
    if field_name in SKILLS_PLURAL_STATE_FIELDS:
        return "_library_skills_" + field_name
    return "_library_skill_" + field_name


@dataclass
class LibrarySkillsState:
    """Every field the Skills subsystem exclusively owns."""

    # Placeholder default only -- see module docstring's "new wrinkle"
    # paragraph: the original `self._library_skill_choice_presented_
    # generation = -1` line is deleted (static literal, safe to fold in),
    # but every field below it that keeps its ORIGINAL line untouched
    # still needs a placeholder here since this object is constructed
    # before those lines run.
    choice_presented_generation: int = -1

    # Placeholder defaults only -- the shared reader-preferences
    # tuple-unpack, the `resolve_adaptive_reader_layout` call, and the
    # persistence-locks dict literal all keep their ORIGINAL `__init__`
    # lines untouched (see module docstring's entangled-trio paragraph),
    # writing through the screen's generated property shim into this
    # object's own fields the instant each original line runs.
    reader_preferences: AdaptiveReaderLayoutPreferences = field(
        default_factory=AdaptiveReaderLayoutPreferences
    )
    reader_layout: AdaptiveReaderEffectiveLayout = field(
        default_factory=lambda: resolve_adaptive_reader_layout(
            0,
            AdaptiveReaderLayoutPreferences(),
            LIBRARY_SKILLS_READER_PROFILE,
        )
    )
    reader_persistence_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    # Placeholder default only -- see module docstring's "new wrinkle"
    # paragraph: the original `self._library_skill_editor_mode =
    # coerce_skill_editor_mode(...)` line keeps running, untouched, at its
    # original position (reads live `library_config` data this
    # constructor call cannot see).
    editor_mode: SkillEditorMode = "basic"
    tool_catalog: tuple[str, ...] = ()
    tool_filter: str = ""
    tool_captured: tuple[str, ...] = ()
    tool_picker_changed: bool = False
    more_actions_open: bool = False
    trust_details_open: bool = False
    mutation_in_flight: bool = False

    # Skills list canvas (Task 3 of the Skills sub-project): sort/filter
    # are pure in-memory operations over the already-fetched
    # ``get_context`` snapshot payload.
    # ``_selected_skill_name`` is recording-only for now -- the
    # in-canvas skill detail/trust editor lands in a later task -- same
    # posture ``handle_library_prompt_row`` originally had before its
    # own editor landed.
    sort: str = "name"
    filter: str = ""
    filter_cursor_context: tuple[int, int] | None = None
    selected_skill_name: str = ""

    # Skill detail/trust editor (Task 4 of the Skills sub-project).
    # Mirrors the prompts editor's own state shape
    # (``_library_prompts_view``/``_library_prompt_detail``/etc.) --
    # ``_selected_skill_name`` (above, Task 3) doubles as the
    # create-vs-update sentinel: ``""`` means "not yet created" (mirrors
    # ``_selected_prompt_id is None``), a real name means "editing an
    # existing skill", routed through by NAME (skills are keyed by name,
    # not a numeric id -- unlike prompts' ``_resolve_editor_prompt_id``
    # complication, ``detail["name"]`` is already the stable identity).
    view: str = "list"

    # Placeholder default only -- see module docstring's "new wrinkle"
    # paragraph: the original `self._library_skill_reader_mode =
    # coerce_skill_reader_mode(None)` line keeps running, untouched, at
    # its original position.
    reader_mode: SkillReaderMode = "overview"

    # task-14902: True while the skills sort chooser's direct-pick
    # strip replaces the list toolbar row (the Notes Sort pattern).
    sort_choices_visible: bool = False

    detail: Mapping[str, Any] | None = None
    detail_generation: int = 0
    detail_loading: bool = False
    detail_error: str = ""
    detail_retryable: bool = False
    original_name: str = ""
    editor_state: SkillEditorState | None = None
    dirty: bool = False
    status: str = ""
    conflict: bool = False

    # Guards against the spurious ``Input.Changed``/``TextArea.Changed``
    # Textual fires when a widget mounts with a non-empty initial value
    # -- same rationale/re-arm timing as ``_library_prompt_editor_armed``.
    editor_armed: bool = False

    # The trust panel's currently-captured review (from
    # ``capture_review``'s result mapping), or ``None`` when no review
    # is active for the open skill. Reset every time a (different)
    # skill is opened -- unlike ``skills_screen.py``'s
    # ``_active_trust_review`` (which persists across row selection
    # within one long-lived screen instance and needs its own
    # staleness reconciliation), this editor always starts a fresh
    # session per open, so no extra staleness check is needed.
    active_review: dict[str, Any] | None = None

    # Task 7 (skills-script-execution): whether the open skill has a
    # standing "always allow scripts" grant, cached here since checking
    # it re-scans the skill's on-disk directory (fingerprint match) and
    # so is only ever read off-thread -- see
    # ``_refresh_library_skill_script_grant``. Reset alongside
    # ``_library_skill_active_review`` on every (re)open; the panel's
    # compose default of "not granted" is corrected in place moments
    # later by that off-thread fetch.
    script_grant: bool = False

    # task-415: inline two-step delete (mirrors
    # ``_library_media_confirming_delete``): the first Delete press
    # arms this; only the recomposed confirm button actually deletes.
    confirming_delete: bool = False

    # task-417: one-shot flag armed by a create-save; the next editor
    # recompose scrolls the action row (Save + status) back into view
    # instead of landing at the top away from what the user pressed.
    scroll_pending: bool = False

    # Task 5 (skills-foundation): the Skills-list header's adaptive
    # trust posture (``SkillTrustService.trust_posture()`` -- Task 3),
    # cached here since it's read off-thread (may touch the OS
    # keyring), never on the compose path. ``""`` hides the header --
    # see ``_refresh_library_skills_trust_posture``.
    trust_posture: str = ""

    # Confirm-gate for the destructive Reset action (wipes the trust
    # manifest -- every skill drops back to needs-review). Armed by
    # either the list header's standalone Reset button or the editor's
    # ``quarantined_manifest_error`` trust panel's own Reset button --
    # both share this single flag/handler set since only one of the
    # two views is ever mounted at a time.
    trust_confirming_reset: bool = False
