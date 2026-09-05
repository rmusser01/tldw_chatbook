"""``LibraryPromptsState`` -- the Prompts subsystem's own fields.

State PR of the Prompts extraction series (wave-6 task 1,
``.superpowers/sdd/2026-09-05-library-decomposition-wave6-prompts``; recipe:
``backlog/docs/library-decomposition-recipe.md``; the skills series --
``library_skills_state.py`` -- is the worked example this mirrors most
closely, since Prompts also splits across THREE shim prefixes and also has
both an entangled reader-preferences trio and a config-computed
``editor_mode``). Every field here was moved verbatim out of
``LibraryScreen.__init__`` in ``tldw_chatbook/UI/Screens/library_screen.py``
-- same default, same type, same comment. ``library_screen.py`` keeps every
original ``_library_prompt_<field>``/``_library_prompts_<field>``/
``_selected_prompt_id`` attribute name alive as a generated getter/setter
``@property`` shim pointing at ``self._prompts_state.<field>`` (a
sentinel-wrapped block right after the ``LibraryScreen`` class body), which
this series' own cleanup PR (task 3) deletes once every remaining screen-side
reference has been retargeted.

**Note on the two ``library_prompts_state`` modules (basename collision).**
This file (``tldw_chatbook/UI/Library_Modules/library_prompts_state.py``) is
the SCREEN's own ``__init__``-field state object. The pre-existing
``tldw_chatbook/Library/library_prompts_state.py`` is a DIFFERENT module --
the Prompts domain layer (``PromptBrowseScope``, ``PromptsListState``,
``PromptEditorState``, ``coerce_prompt_editor_mode``, ...), consumed by
``LibraryScreen._build_library_prompts_state`` and by the prompts canvas
widgets, and covered by its own ``Tests/Library/test_library_prompts_
state.py``. Neither module imports the other by bare name; every reference
in the tree is package-qualified, so the shared basename is inert. This
exact pair already exists for six other subsystems (``library_conversations_
state.py``, ``library_export_state.py``, ``library_skills_state.py``,
``library_ingest_state.py``, ...) -- it is the established directory
precedent, not a new pattern.

Three prefixes, not two -- the plan's own "two prefix families" framing was
the floor, not the ceiling, and the recipe's own "startswith enumeration
trap" lesson (§11) is why the ownership census was run as a SUBSTRING match
on "prompt" over every ``__init__``-stored attribute rather than a
``startswith`` filter on the two ``_library_prompt(s)_`` prefixes:

- ``_library_prompt_`` (singular) is the DEFAULT prefix -- 31 of the 43
  fields (the editor/detail/delete-batch cluster: ``dirty``, ``status``,
  ``detail``, ``delete_pending_targets``, ...).
- ``_library_prompts_`` (plural) covers 11 fields describing the prompts
  LIST/browse/import surface as a whole rather than one prompt's own editor
  state: ``view``, ``debounce_timer``, ``filter_cursor_context``,
  ``sort_choices_visible``, ``mutation_in_flight``, the import trio, and the
  reader trio. ``PROMPTS_PLURAL_STATE_FIELDS`` is the single authoritative
  home for that set.
- ``_`` (bare underscore, no "prompt(s)" word in the prefix position at all)
  covers exactly ONE field: ``selected_prompt_id`` (the original attribute
  is ``_selected_prompt_id``) -- the same third-prefix shape the skills
  series found in ``_selected_skill_name``.

``prompt_state_shim_attr()`` below is the single function BOTH the screen's
shim-generator loop and this subsystem's wiring test call to resolve a field
name to its original attribute name, rather than each independently
recomputing the three-way branch (the drift risk a duplicated frozenset
would reintroduce).

Field ownership (recipe §2 script, substring "prompt" over
``__init__``-stored attributes): **46 fields** found, of which **43 MOVE**,
**3 are WIRING** and **0 are BLOCKED**.

The 3 WIRING fields are NOT part of this dataclass at all -- exactly the
``_conversation_reader_controller``/``_library_collections_capture_
controller``/``_library_skill_import_coordinator`` "capture-controller"
precedent. Each holds a live controller instance constructed with lambdas
that close over the screen, and each stays a plain ``LibraryScreen``
attribute at its original ``__init__`` position, untouched by this move:

- ``_library_prompt_history_controller`` (``LibraryPromptHistoryController``)
- ``_library_prompt_browse_controller`` (``LibraryPromptBrowseController`` --
  the prior-extracted browse WIRING the plan named in advance)
- ``_library_prompt_collections_controller``
  (``LibraryPromptCollectionsController``)

**Zero fields are BLOCKED by the >=2-subsystems rule**, but two need their
cross-subsystem readers recorded rather than waved past, since a name-based
tagging heuristic alone would have mis-tagged both (the recipe's own caveat
that the script's tags are name-based, not body-based):

- ``mutation_in_flight`` (plural) is READ by 93 methods, 16 of them not
  prompt-named. Thirteen are shell/plumbing navigation guards
  (``apply_navigation_context``, ``handle_library_rail_row``,
  ``_select_library_rail_row``, ``flush_pending_work``, ``compose_content``,
  ...). The other three are Notes-named (``_show_library_file_notes``,
  ``_show_library_database_notes``, ``_return_to_library_database_notes``)
  and use the IDENTICAL read-only guard shape -- ``if self._library_prompts_
  mutation_in_flight: return`` -- i.e. they consult a Prompts busy flag
  before leaving/entering a Notes surface, exactly as the thirteen shell
  guards do. Ownership is unambiguous: all FOUR writers are prompt-named
  (``_delete_library_prompts``, ``_settle_library_prompt_delete``,
  ``_undo_library_prompt_delete``, ``handle_library_prompt_delete_undo``),
  and no Notes method ever writes it. This matches the search+RAG series'
  own already-landed precedent for ``_rag_search_state.query``, which
  ``_show_library_file_notes`` likewise reads without owning.
- ``reader_layout`` (plural) is read by ``_toggle_library_media_reader_pane``
  -- the generic multi-subsystem pane dispatcher whose name merely contains
  another subsystem's prefix. Collections, conversations and skills all
  already moved their identically-named field past this same reader, and
  this file follows them.

Four fields keep their *original* ``__init__`` assignment line completely
untouched, because ``self._prompts_state`` must be constructed EARLY -- right
after ``self._skills_state`` and BEFORE the shared reader-preferences
tuple-unpack -- for the same reason the collections/skills series document,
and a field whose original line runs after that forced construction point
cannot be passed as a constructor argument:

- ``reader_preferences`` is one of eight targets unpacked from a single
  ``self._load_library_reader_preference_snapshot()`` call shared with Media,
  Conversations, Notes, File Notes, Collections and Skills.
- ``reader_layout`` is derived from ``reader_preferences`` (via
  ``resolve_adaptive_reader_layout``) after that unpack settles.
- ``reader_persistence_locks`` depends on a local ``asyncio.Lock()``
  (``library_pane_persistence_lock``) also reused by five other subsystems'
  persistence-lock dicts.
- ``editor_mode`` calls ``coerce_prompt_editor_mode`` with live
  ``library_config`` data this constructor cannot see -- the skills series'
  own ``editor_mode``/``reader_mode`` "new wrinkle" shape exactly.

All four dataclass defaults below are therefore momentary placeholders,
overwritten through the generated property shim before anything else reads
them. Every OTHER field's original line is deleted outright: 39 static
literals plus two pure no-argument factory calls folded into
``default_factory`` (``PromptSelectionBasket`` and ``local_prompt_
capabilities`` -- both read only module constants and have no side effects,
so calling them at the earlier construction point is behaviorally
transparent, matching Ingest's own ``LibraryIngestFormState()``/
``threading.Lock()`` folds).
"""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from textual.timer import Timer
from textual.widget import Widget

from ...Library.library_prompts_state import (
    PromptEditorMode,
    PromptEditorState,
    PromptSelectionBasket,
    PromptSelectionEntry,
)
from ...Prompt_Management.prompt_batch_models import (
    PromptBatchDeleteResult,
    PromptBatchTarget,
)
from ...Prompt_Management.prompt_source_capabilities import (
    PromptSourceCapabilities,
    local_prompt_capabilities,
)
from ...Utils.adaptive_reader_state import (
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    resolve_adaptive_reader_layout,
)
from ...Widgets.Prompts.prompt_block_editor_state import PromptBlockEditorState
from .screen_constants import LIBRARY_PROMPTS_READER_PROFILE

#: See the module docstring's three-prefix note: the single authoritative
#: home for which field names use the plural ``_library_prompts_`` shim
#: prefix instead of the cluster's default singular ``_library_prompt_``
#: prefix.
PROMPTS_PLURAL_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "debounce_timer",
        "filter_cursor_context",
        "view",
        "sort_choices_visible",
        "mutation_in_flight",
        "import_open",
        "import_path",
        "import_status",
        "reader_preferences",
        "reader_layout",
        "reader_persistence_locks",
    }
)

#: The single field using the bare ``_`` prefix (no "prompt"/"prompts" word
#: in the prefix position of the original attribute name).
PROMPT_UNPREFIXED_STATE_FIELDS: frozenset[str] = frozenset({"selected_prompt_id"})


def prompt_state_shim_attr(field_name: str) -> str:
    """The original ``LibraryScreen`` attribute name for a state field.

    Single-source resolution of the three-way prefix mapping documented in
    this module's own docstring -- used by BOTH the screen's generated shim
    loop and this subsystem's wiring test, so the mapping cannot drift into
    two independently-typed copies.

    Args:
        field_name: A ``LibraryPromptsState`` dataclass field name.

    Returns:
        The flat ``LibraryScreen`` attribute name that field shims under.
    """
    if field_name in PROMPT_UNPREFIXED_STATE_FIELDS:
        return "_" + field_name
    if field_name in PROMPTS_PLURAL_STATE_FIELDS:
        return "_library_prompts_" + field_name
    return "_library_prompt_" + field_name


@dataclass
class LibraryPromptsState:
    """Every field the Prompts subsystem exclusively owns."""

    debounce_timer: Timer | None = None
    filter_cursor_context: tuple[int, int] | None = None
    select_mode: bool = False
    selection: PromptSelectionBasket = field(default_factory=PromptSelectionBasket)
    selected_prompt_id: int | None = None
    view: str = "list"

    # task-14902: True while the prompts sort chooser's direct-pick
    # strip replaces the list toolbar row (the Notes Sort pattern).
    sort_choices_visible: bool = False
    detail: Mapping[str, Any] | None = None
    loaded_id: int | None = None
    detail_generation: int = 0
    detail_loading: bool = False
    detail_selected_name: str = ""
    detail_error: str = ""
    detail_retryable: bool = False
    original_name: str = ""
    version: int | None = None
    dirty: bool = False
    status: str = ""
    conflict_snapshot: PromptEditorState | None = None
    block_state: PromptBlockEditorState | None = None

    # Placeholder default only -- see the module docstring's
    # forced-early-construction paragraph: the original
    # `self._library_prompt_editor_mode = coerce_prompt_editor_mode(...)`
    # line keeps running, untouched, at its original position (it reads
    # live `library_config` data this constructor call cannot see).
    editor_mode: PromptEditorMode = "basic"

    # Explicit provenance for an unsaved canonical structured copy.
    # Legacy block edits can clear both lane origins, so origins cannot
    # truthfully distinguish conversion/duplication from ordinary edits.
    detached_structured: bool = False
    capabilities: PromptSourceCapabilities = field(
        default_factory=local_prompt_capabilities
    )
    include_starter_content: bool = False
    delete_pending_fingerprint: str | None = None
    delete_inflight_fingerprint: str | None = None
    mutation_generation: int = 0
    delete_pending_targets: tuple[PromptBatchTarget, ...] | None = None
    delete_pending_entries: tuple[PromptSelectionEntry, ...] | None = None
    delete_pending_selection_generation: int | None = None
    delete_pending_editor_prompt_id: int | None = None

    # TASK-15101 / ADR-055: delete and Undo both mutate the exact Prompt
    # browse result, rail count, and receipt. Admission is shared so the
    # two directions cannot interleave through separate worker groups.
    mutation_in_flight: bool = False
    mutation_status: str = ""
    delete_receipt: PromptBatchDeleteResult | None = None
    mutation_disabled_states: dict[Widget, bool] = field(default_factory=dict)

    # Task 8b Fix wave 1 (Minor): the exact name that triggered the
    # current "name-in-use" status, captured at the moment that status
    # is set -- NOT re-derived from the live Name field at "Open
    # existing" time, which can have drifted (the user can keep typing
    # after a failed Save without re-saving) from the name that
    # actually collided. See ``_open_library_prompt_colliding_with_current_name``.
    name_in_use: str = ""

    # Toolbar Import… state (Task 5): a path Input (file OR folder)
    # inlined below the sort/Import…/Export… toolbar, worker-executed
    # on Run/Enter. See ``_run_library_prompts_import``.
    import_open: bool = False
    import_path: str = ""
    import_status: str = ""

    # Guards against the spurious ``Input.Changed``/``TextArea.Changed``
    # Textual fires when a widget mounts with a non-empty initial value
    # -- without this, opening a prompt would immediately mark it dirty
    # even though the user never typed anything. Re-armed via
    # ``call_after_refresh`` after every prompt-editor (re)compose,
    # mirroring ``_library_note_editor_armed``.
    editor_armed: bool = False

    # Placeholder defaults only -- the shared reader-preferences
    # tuple-unpack, the `resolve_adaptive_reader_layout` call, and the
    # persistence-locks dict literal all keep their ORIGINAL `__init__`
    # lines untouched (see the module docstring's forced-early-construction
    # paragraph), writing through the screen's generated property shim into
    # this object's own fields the instant each original line runs.
    reader_preferences: AdaptiveReaderLayoutPreferences = field(
        default_factory=AdaptiveReaderLayoutPreferences
    )
    reader_layout: AdaptiveReaderEffectiveLayout = field(
        default_factory=lambda: resolve_adaptive_reader_layout(
            0,
            AdaptiveReaderLayoutPreferences(),
            LIBRARY_PROMPTS_READER_PROFILE,
        )
    )
    reader_persistence_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
