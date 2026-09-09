"""``LibraryNotesState`` -- the Notes subsystem's own fields.

State PR of the Notes extraction series (wave-8 task 1,
``.superpowers/sdd/2026-09-08-library-decomposition-wave8-notes``; recipe:
``backlog/docs/library-decomposition-recipe.md``; the media series --
``library_media_state.py`` -- is the worked example this mirrors most closely,
since Notes likewise carries a bare-underscore extra-prefix field and an
entangled reader-preferences group, and additionally carries a SECOND
reader-preferences destination of its own). Every field here was moved verbatim
out of ``LibraryScreen.__init__`` in ``tldw_chatbook/UI/Screens/
library_screen.py`` -- same default, same type, same comment. The state PR KEPT
every original ``_library_notes_<field>``/``_library_note_<field>``/
``_library_file_notes_<field>``/``_selected_note_id`` attribute name alive as a
generated getter/setter ``@property`` shim on ``LibraryScreen``, between
sentinel comments, so no method body had to be edited; task 3 (the cleanup PR)
retargeted every one of those references to ``self._notes_state.<field>`` and
DELETED that screen block -- the identical generated loop lives on
``LibraryNotesController`` now, one layer down, and is what keeps the moved
bodies byte-for-byte.

**Note on the two ``library_notes_state`` modules (basename collision).**
This file (``tldw_chatbook/UI/Library_Modules/library_notes_state.py``) is the
SCREEN's own ``__init__``-field state object. The pre-existing
``tldw_chatbook/Library/library_notes_state.py`` is a DIFFERENT module -- the
Notes domain layer (``LibraryNoteDeleteReceipt``, ``LibraryNotesFocusIdentity``,
``LibraryNotesOperationState``, ``build_library_notes_list_state``, ...),
consumed by ``LibraryScreen`` and the notes canvas widgets and covered by its
own ``Tests/Library/test_library_notes_state.py``. This file imports FROM that
one (package-qualified, ``...Library.library_notes_state``); nothing anywhere
imports either by bare name, so the shared basename is inert. The identical
pair already exists for media and prompts and, less exactly, for six other
subsystems -- it is the established directory precedent, not a new pattern.

FOUR prefix families, not two. The ownership census was run as a SUBSTRING
match on "note" over every ``__init__``-stored attribute (plus a full
class-body ``Assign``/``AnnAssign`` scan, which found nothing -- unlike Media,
Notes has no class-level-only attribute) rather than a ``startswith`` filter,
per the conversations exemplar's own "startswith enumeration trap" lesson:

- ``_library_notes_`` is the DEFAULT prefix -- 73 of the 100 moved fields.
- ``_library_note_`` (SINGULAR) covers 21 fields, listed in
  ``NOTE_SINGULAR_STATE_FIELDS`` below -- the same singular/plural split the
  prompts and skills series each carried, with the majority family reversed
  (Prompts' default was the singular one).
- ``_library_file_notes_`` covers 5 fields, listed in
  ``FILE_NOTES_STATE_FIELDS``. Unlike the other two families this one's
  dataclass field names KEEP their ``file_notes_`` marker
  (``file_notes_reader_preferences``, not ``reader_preferences``), because
  stripping it would collide head-on with the Database-Notes trio of the same
  name: Notes owns TWO reader destinations (``"notes"`` and ``"notes_files"``
  in ``_replace_library_reader_preference``'s own dispatch dict), so
  ``reader_preferences``/``reader_layout``/``reader_persistence_locks`` each
  exist twice. The mapping for this family is therefore ``"_library_" +
  field_name``.
- ``_`` (bare underscore, no "note" word in the prefix position at all) covers
  exactly ONE field: ``selected_note_id`` (the original attribute is
  ``_selected_note_id``) -- the same fourth-prefix shape the skills series
  found in ``_selected_skill_name``, the prompts series in
  ``_selected_prompt_id`` and the media series in ``_selected_media_id``.

``notes_state_shim_attr()`` below is the single function BOTH the screen's
shim-generator loop and this subsystem's wiring test call to resolve a field
name to its original attribute name, rather than each independently
recomputing the branch (the drift risk a duplicated frozenset would
reintroduce).

Field ownership (recipe §2 script, substring "note" over ``__init__``-stored
attributes, plus the class-body scan): **105 attributes** found, of which
**100 MOVE**, **3 are WIRING** and **2 are BLOCKED**.

The 3 WIRING attributes are NOT part of this dataclass at all -- exactly the
``_conversation_reader_controller``/``_library_media_browse_controller``
"prior-extracted controller" precedent. Each holds a live coordinator instance
constructed with callables that close over the screen, and each stays a plain
``LibraryScreen`` attribute at its original ``__init__`` position, untouched by
this move:

- ``_library_note_import_controller`` (``LibraryNoteImportController``, from
  ``library_note_import_controller.py``)
- ``_library_notes_sync_controller`` (``LibraryNotesSyncController``, from
  ``library_notes_sync_controller.py``)
- ``_library_note_session`` (``DatabaseNoteSessionCoordinator``, constructed
  over the ``_LibraryDatabaseNoteSessionPort`` from ``note_session_port.py``,
  whose ``run_service_call=self._run_library_service_call`` closes over the
  screen)

Those are three of the four prior-extracted Notes wiring modules the wave-8
plan named. The FOURTH, ``library_notes_work_session.py``, is different in kind
and is NOT wiring: it exports a pure reducer plus two enums, and the screen's
two work-session attributes hold DATA (a ``NotesWorkSessionPhase`` member and a
bool), not a live object, so ``work_session_phase`` and
``work_session_activation_pending`` MOVE like any other field. Their
``_SCREEN_FIELDS`` string-tuple consumer in ``library_inspection_admission.py``
keeps resolving through the generated shim for the whole state PR; retargeting
it (the same possibly-dotted-path shape ``_assign_library_reader_preferences_
attribute`` already solves for four subsystems) is the cleanup PR's job.

The 2 BLOCKED fields are ``_library_notes_programmatic_focus_target`` and
``_library_notes_restoring_focus``. Both carry a notes name, and Notes' own
methods do write them -- but so does MEDIA, from bodies that have already left
this screen: ``library_media_controller.py`` binds each with a getter **and a
setter** accessor and lists them, by name, among the "5 shared shell state
[names] this cluster also WRITES" in its own module docstring (beside
``_library_selected_row_id``, the recipe's canonical >=2-subsystems example).
A field a second subsystem WRITES is shared shell state by the recipe's own
rule, so they stay screen-owned; the screen-resident writers
``_commit_library_media_return``, ``_sync_library_media_browse_state``,
``_sync_library_media_trash_state`` and ``_focus_library_list_entry`` confirm
the same verdict from the other side.

**The line this series draws, and why it is not stricter.** Five notes-named
fields are referenced from another subsystem's already-moved bodies. The two
above are WRITTEN there and are BLOCKED. The other three --
``_library_notes_compact``, ``_library_notes_source`` and
``_library_notes_focus_intent_generation`` -- are only READ there (media binds
all three getter-only; ``library_conversation_reader_controller.py`` binds
``focus_intent_generation`` getter-only as well), and every WRITER of each is
either Notes' own code or the shell (``__init__``, ``on_descendant_focus``,
``_transition_library_notes_presentation``, ``_set_library_notes_source``,
``_return_to_library_database_notes``). They MOVE, on the landed precedent of
``_library_prompts_mutation_in_flight`` -- moved by the prompts series into
``LibraryPromptsState`` and read to this day by three Notes-named screen
methods (``_show_library_file_notes``, ``_show_library_database_notes``,
``_return_to_library_database_notes``) plus one ``__init__`` binding lambda --
and of the media series' own ruling for ``view``, read by a Notes-named guard
without being owned by it. A cross-subsystem READ is a route guard; a
cross-subsystem WRITE is co-ownership.

**Shell/plumbing readers and writers recorded rather than waved past** (the
recipe's caveat that the census's tags are name-based, not body-based):

- ``library_unavailable_navigation.py`` -- a module of free functions taking
  the screen as ``self``, not a subsystem -- reads and writes 15 of these
  fields directly. It is shell/plumbing (unavailable-destination navigation is
  not one of the recipe's eleven subsystems), so per §2 those references move
  with Notes and the cleanup PR retargets them.
- ``canvas_sync.py``'s ``"notes"`` branch reads ``screen._library_note_session``
  (a WIRING attribute, so nothing changes for it) and hands
  ``partial(getattr, screen, "_library_notes_focus_intent_generation")`` to the
  canvas -- a by-string read that keeps resolving through the shim.
- The same file's ``f"_library_{kind}_row_selection"`` (recipe §3's FIFTH
  census spelling) composes ``_library_notes_row_selection`` for
  ``kind == "notes"``. It resolves through the shim for the whole state PR;
  the dotted branch and its mutation-verified guard belong to whichever commit
  DELETES the shim, exactly as the conversations and media branches there did.
- ``_replace_library_reader_preference``/``_persist_library_reader_preference``
  and ``_close_open_library_choice_strip`` dispatch by attribute-name string
  through ``_assign_library_reader_preferences_attribute``; the ``"notes"`` and
  ``"notes_files"`` rows keep working through the shim.
- The background notes-sync runtime does NOT write any of these fields off the
  UI thread: the two publish callbacks the screen hands the controllers
  (``_publish_library_note_import_snapshot``,
  ``_publish_library_notes_lasting_sync_snapshot``) both touch the DOM
  (``_sync_library_canvas`` / ``_apply_library_notes_footer_context``) and are
  therefore already UI-thread-only. A property shim is in any case exactly as
  atomic as the plain attribute store it replaces -- one store on one object --
  so this move introduces no new cross-thread hazard.

**The four-member shell family, checked and ruled INAPPLICABLE.** The wave-8
plan carried forward media's 1 BLOCKED field as one of a four-member shell
family and warned that "the notes twin stays screen-owned for the same
reason". Re-derived against the live file, the family is
``_library_pending_list_entry_focus``, ``_library_list_entry_focus_generation``,
``_library_pending_list_entry_media_return`` and
``_library_pending_list_entry_focus_anchor`` -- assigned together in
``_arm_library_list_entry_focus`` and cleared together in
``_disarm_library_list_entry_focus``. There is NO notes member: only the media
one is subsystem-named, and none of the four contains "note", so none is in
this wave's candidate set at all. The family stays screen-owned and whole, as
it already was; nothing in this PR touches it.

**The suspend/resume seam (dev's TASK-31521), ruled field by field.**
``on_screen_suspend`` reaches ``_library_notes_autosave_timer`` through a
FLAT-NAME STRING LOOP (``for attr in ("_library_notes_autosave_timer",
"_library_source_snapshot_timeout_timer")``: ``getattr`` then ``setattr``).
Both halves keep working through the generated shim, so this PR needs no edit
there and makes none -- the ingest, prompts and media timers each got their
explicit ``self._<subsystem>_state.<timer>`` block in the commit that DELETED
their shim, not in the commit that moved the field, and
``Tests/UI/test_library_screen_reuse.py``'s own comments say so in as many
words ("the screen's generated shim block was deleted in the X cleanup PR, so
a ``getattr`` on the old flat name passes VACUOUSLY"). Wave-8 task 3 owns that
block and the matching retarget of both assertions in that file. The other
lifecycle seam, ``_refresh_library_visit_surfaces`` (the ``on_screen_resume``
dispatch), reads ``source``/``view``/``stage`` and writes nothing -- ordinary
shell/plumbing reads that move and are retargeted at cleanup. The
screen-lifecycle fields themselves (``_library_screen_suspended``,
``_library_visit_entered``) are not notes-named, are not in this census, and
stay screen-owned.

**One pre-existing red this move flips GREEN, disclosed rather than banked.**
``Tests/Notes/test_notes_sync_cutover.py::test_library_screen_has_no_legacy_
timer_worker_or_mutating_handler`` AST-walks ``library_screen.py`` for any
``ast.Attribute`` whose name starts with ``_library_notes_auto_sync_timer`` and
fails today on the single ``__init__`` assignment at the parent
(``889e12b86``: ``assert ['_library_notes_auto_sync_timer:3580'] == []``).
Folding that field into this dataclass removes the only literal spelling from
that file, so the guard passes afterwards -- for a reason the guard did not
intend, since the field still exists, one indirection away. It is recorded here
and in the task report so the next reader does not mistake a vacuous pass for a
repair; retargeting that census (its own fifth-spelling problem: the flat name
is composed, not spelled, inside the shim loop) belongs with the cutover work
that owns it, not with a pure field move.

NINE fields keep their *original* ``__init__`` assignment line completely
untouched, because ``self._notes_state`` must be constructed EARLY -- at the
position of the first removed field, well above the shared reader-preferences
tuple-unpack -- for the same reason the collections/skills/prompts/media series
document, and a field whose original line runs after that forced construction
point cannot be passed as a constructor argument:

- ``file_notes_workspace_factory`` is assigned in an ``if``/``else`` whose first
  branch DEFINES a closure over ``app_instance``; a statement block cannot be
  folded into a default.
- ``reader_preferences`` and ``file_notes_reader_preferences`` are two of eight
  targets unpacked from a single ``self._load_library_reader_preference_
  snapshot()`` call shared with Conversations, Media, Collections, Prompts and
  Skills.
- ``reader_layout`` and ``file_notes_reader_layout`` are derived from those two
  (via ``resolve_adaptive_reader_layout``) after the unpack settles.
- ``reader_persistence_locks`` and ``file_notes_reader_persistence_locks``
  depend on a local ``asyncio.Lock()`` (``library_pane_persistence_lock``) also
  reused by five other subsystems' persistence-lock dicts.
- ``import_snapshot`` and ``lasting_sync_snapshot`` read
  ``.presentation_snapshot``/``.snapshot`` off the two WIRING controllers,
  which are themselves constructed far below.

Their dataclass defaults below are therefore momentary placeholders,
overwritten by their own original ``__init__`` lines (which route through the
generated shim) before anything else reads them. Two of the nine --
``import_snapshot`` and ``lasting_sync_snapshot`` -- take ``None`` as that
placeholder and widen their annotation to ``| None`` accordingly, because
``LibraryNoteImportSnapshot`` and ``LibraryNotesLastingSyncSnapshot`` require
12 and 6 constructor arguments respectively (verified against the live classes),
so no cheap side-effect-free stand-in exists the way it did for
``AdaptiveReaderLayoutPreferences()``.

ZERO fields become constructor arguments: every remaining default is a static
literal or a pure no-argument/one-literal-argument factory call, so
``LibraryNotesState()`` takes no arguments at all -- unlike Media's three. The
91 fields whose original line is deleted outright break down as **86** static
literals or empty collection displays and **5** factory calls folded into
``default_factory``: ``RowSelection("notes")``, ``set()``, ``frozenset()``
(twice) and the two-key ``{"database": None, "files": None}`` dict display.
``RowSelection`` only stores its argument and an empty set; the rest are
builtins; none has a side effect, so calling them at the earlier construction
point is behaviorally transparent, matching Media's own folds.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from textual.timer import Timer
from textual.widget import Widget

from ...Library.library_note_import_state import LibraryNoteImportSnapshot
from ...Library.library_notes_lasting_sync_state import (
    LibraryNotesLastingSyncSnapshot,
)
from ...Library.library_notes_state import (
    LibraryNoteDeleteReceipt,
    LibraryNotesFocusIdentity,
    LibraryNotesOperationState,
    LibraryNotesTrashState,
)
from ...Library.library_notes_tree_paging import (
    NotesBranchKey,
    NotesBranchSliceState,
)
from ...Library.library_notes_tree_state import (
    LibraryNotesFilterState,
    LibraryNotesTreeReceipt,
)
from ...Library.row_selection import RowSelection
from ...Utils.adaptive_reader_state import (
    AdaptiveReaderEffectiveLayout,
    AdaptiveReaderLayoutPreferences,
    resolve_adaptive_reader_layout,
)
from .library_notes_work_session import NotesWorkSessionPhase
from .screen_constants import (
    LIBRARY_FILE_NOTES_READER_PROFILE,
    LIBRARY_NOTES_READER_PROFILE,
    LIBRARY_NOTES_SOURCE_DATABASE,
)
from .screen_support_types import _LibraryNotesDeletedFolderReceipt

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Kept type-only, exactly as `library_screen.py` keeps it: the Folder
    # Files workspace widget is constructed lazily on first open, and a
    # runtime import here would drag it into the first-paint module census.
    from ...Widgets.Library.library_file_notes_workspace import (
        LibraryFileNotesWorkspace,
    )

#: The 21 fields whose original attribute uses the SINGULAR
#: ``_library_note_`` prefix rather than the plural default.
NOTE_SINGULAR_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "autosave_generation",
        "autosave_state",
        "confirming_delete",
        "context",
        "create_counter",
        "create_running",
        "create_status",
        "create_token",
        "delete_origin_context",
        "delete_origin_preview",
        "delete_receipt",
        "editor_armed",
        "import_snapshot",
        "load_message",
        "load_state",
        "pending_blank_gc_id",
        "presentation_syncing",
        "preview",
        "session_blank_id",
        "shortcut_status",
        "title_user_edited",
    }
)

#: The 5 Folder-Files fields, whose original attribute uses the
#: ``_library_file_notes_`` prefix. These field names KEEP the
#: ``file_notes_`` marker (see the module docstring): Notes owns two reader
#: destinations, so three of the five would otherwise collide with the
#: Database-Notes fields of the same stripped name.
FILE_NOTES_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "file_notes_reader_layout",
        "file_notes_reader_persistence_locks",
        "file_notes_reader_preferences",
        "file_notes_workspace",
        "file_notes_workspace_factory",
    }
)

#: The single field using the bare ``_`` prefix (no "note" word in the
#: prefix position of the original attribute name).
NOTE_UNPREFIXED_STATE_FIELDS: frozenset[str] = frozenset({"selected_note_id"})


def notes_state_shim_attr(field_name: str) -> str:
    """The original ``LibraryScreen`` attribute name for a state field.

    Single-source resolution of the four-way prefix mapping documented in this
    module's own docstring -- used by BOTH ``LibraryScreen``'s generated shim
    loop and this subsystem's wiring test, so the mapping cannot drift into
    two independently-typed copies.

    Args:
        field_name: A ``LibraryNotesState`` dataclass field name.

    Returns:
        The flat original ``LibraryScreen`` attribute name that field shims
        under.
    """
    if field_name in NOTE_UNPREFIXED_STATE_FIELDS:
        return "_" + field_name
    if field_name in FILE_NOTES_STATE_FIELDS:
        return "_library_" + field_name
    if field_name in NOTE_SINGULAR_STATE_FIELDS:
        return "_library_note_" + field_name
    return "_library_notes_" + field_name


@dataclass
class LibraryNotesState:
    """Every field the Notes subsystem exclusively owns."""

    source: Literal["database", "files"] = LIBRARY_NOTES_SOURCE_DATABASE
    work_session_phase: NotesWorkSessionPhase = NotesWorkSessionPhase.INACTIVE
    work_session_activation_pending: bool = False
    file_notes_workspace: LibraryFileNotesWorkspace | None = None

    # Placeholder default only -- see the module docstring's
    # forced-early-construction paragraph: the original `if`/`else` (whose
    # first branch defines a closure over `app_instance`) keeps running,
    # untouched, at its original position.
    file_notes_workspace_factory: Callable[[], LibraryFileNotesWorkspace] | None = None

    # Placeholder defaults only -- see the module docstring's
    # forced-early-construction paragraph: the shared
    # `_load_library_reader_preference_snapshot()` tuple-unpack, the two
    # `resolve_adaptive_reader_layout` calls derived from it and the two
    # persistence-lock dict literals (which share the local
    # `library_pane_persistence_lock`) all keep their ORIGINAL `__init__`
    # positions, writing into this object's own fields the instant each
    # original line runs.
    reader_preferences: AdaptiveReaderLayoutPreferences = field(
        default_factory=AdaptiveReaderLayoutPreferences
    )
    file_notes_reader_preferences: AdaptiveReaderLayoutPreferences = field(
        default_factory=AdaptiveReaderLayoutPreferences
    )
    reader_layout: AdaptiveReaderEffectiveLayout = field(
        default_factory=lambda: resolve_adaptive_reader_layout(
            0,
            AdaptiveReaderLayoutPreferences(),
            LIBRARY_NOTES_READER_PROFILE,
        )
    )
    file_notes_reader_layout: AdaptiveReaderEffectiveLayout = field(
        default_factory=lambda: resolve_adaptive_reader_layout(
            0,
            AdaptiveReaderLayoutPreferences(),
            LIBRARY_FILE_NOTES_READER_PROFILE,
        )
    )
    reader_persistence_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    file_notes_reader_persistence_locks: dict[str, asyncio.Lock] = field(
        default_factory=dict
    )

    view: str = "list"
    lasting_origin: str | None = None
    select_mode: bool = False
    row_selection: RowSelection = field(
        default_factory=lambda: RowSelection("notes")
    )
    sort: str = "newest"
    sort_choices_visible: bool = False
    filter: str = ""
    filter_records: list | None = None
    filter_generation: int = 0
    tree_filter_state: LibraryNotesFilterState | None = None
    filter_navigation_generation: int | None = None
    navigation_status: str = ""
    notice: str = ""
    tree_expanded_ids: set[str] = field(default_factory=set)
    tree_branches: dict[NotesBranchKey, NotesBranchSliceState] = field(
        default_factory=dict
    )
    tree_topology_epoch: int = 0
    tree_lifecycle_generation: int = 0
    tree_request_generations: dict[NotesBranchKey, int] = field(default_factory=dict)
    tree_navigation_requests: dict[NotesBranchKey, int] = field(default_factory=dict)
    tree_target_offsets: dict[NotesBranchKey, int] = field(default_factory=dict)
    tree_status_by_slice: dict[NotesBranchKey, dict[str, tuple[int, str]]] = field(
        default_factory=dict
    )
    tree_status_revision: int = 0
    tree_protected_folder_ids: frozenset[str] = field(default_factory=frozenset)
    tree_inactive_managed_folder_ids: frozenset[str] = field(
        default_factory=frozenset
    )
    tree_selected_placement_id: str = ""
    tree_pending_target_placement_id: str = ""
    filter_browse_receipt: LibraryNotesTreeReceipt | None = None
    deleted_folder_receipt: _LibraryNotesDeletedFolderReceipt | None = None
    create_counter: int = 0
    create_token: str | None = None
    create_running: bool = False
    create_status: str = ""

    # TASK-15100 / ADR-055: create, delete, and Undo all mutate the same
    # cached Notes rows/count/receipt. One admission flag keeps those
    # writes serialized instead of letting separate workers race.
    mutation_in_flight: bool = False
    delete_receipt: LibraryNoteDeleteReceipt | None = None

    # task-32144: the standing second safety net behind that receipt -- the
    # soft-deleted page plus its exact total, reloaded on a fresh Notes visit
    # and after every delete/restore. ``None`` means "not read yet", which is
    # what keeps the "Recently deleted (N)" row off a failed read.
    trash: LibraryNotesTrashState | None = None
    operation_counter: int = 0
    operation: LibraryNotesOperationState | None = None

    # Placeholder defaults only -- see the module docstring's
    # forced-early-construction paragraph: both original lines read a
    # snapshot off a WIRING controller constructed far below, and neither
    # snapshot class is no-argument constructible, so `None` stands in until
    # the original line runs.
    import_snapshot: LibraryNoteImportSnapshot | None = None
    lasting_sync_snapshot: LibraryNotesLastingSyncSnapshot | None = None
    selected_note_id: str = ""
    load_state: str = "idle"
    load_message: str = ""
    autosave_state: str = "idle"
    autosave_timer: Timer | None = None
    autosave_generation: int = 0
    confirming_delete: bool = False
    preview: bool = False
    context: bool = False
    delete_origin_context: bool = False
    delete_origin_preview: bool = False

    # Task 7 owns measured breakpoint transitions. Task 5 consumes this
    # explicit presentation input now so compact/wide utility grouping is
    # testable without coupling the canvas to terminal geometry.
    compact: bool = False
    stage: Literal["rail", "notes"] = "rail"

    # TASK-23151: the resize legs' stage-visibility call runs ABOVE the
    # compact-crossing early-out, because the emergency band (64 cells)
    # is a different band from the compact breakpoint (120) and a 63<->64
    # crossing must still re-apply geometry. This records the signature
    # ``_apply_library_notes_stage_visibility`` last settled -- refreshed
    # by that function on EVERY call, so no seam can leave it stale --
    # and a resize frame matching it skips the leg entirely.
    stage_applied_signature: tuple[Any, ...] | None = None
    explicit_stage_intent: bool = False
    pending_focus_identity: LibraryNotesFocusIdentity | None = None
    pending_focus_waits_for_snapshot: bool = False
    navigation_generation: int = 0
    pending_focus_generation: int | None = None
    responsive_focus_memory: LibraryNotesFocusIdentity | None = None
    last_presented_focus: LibraryNotesFocusIdentity | None = None
    pre_resize_focus: LibraryNotesFocusIdentity | None = None
    interaction_focus: LibraryNotesFocusIdentity | None = None
    resize_epoch: int = 0
    resize_settling: bool = False
    scroll_intent_generation: int = 0
    transition_scroll_generation: int = 0
    focus_intent_generation: int = 0
    transition_focus_generation: int = 0
    authority_focus: dict[Literal["database", "files"], Widget | None] = field(
        default_factory=lambda: {"database": None, "files": None}
    )
    last_user_scroll_focus: LibraryNotesFocusIdentity | None = None
    last_user_focus: LibraryNotesFocusIdentity | None = None
    browse_return_receipt: LibraryNotesTreeReceipt | None = None
    recompose_generation: int = 0
    shortcut_status: str = ""
    presentation_syncing: bool = False

    # Guards against the spurious ``Input.Changed`` that Textual fires
    # when an ``Input(value=...)`` widget mounts with a non-empty
    # initial value: without this, opening a note (or leaving a
    # conflict) would immediately mark the note dirty and arm an
    # autosave even though the user never typed anything. Re-armed via
    # ``call_after_refresh`` after every notes-editor (re)compose.
    editor_armed: bool = False

    # LIB-14: display-only flag for a note created via "Blank note"
    # that has not been touched YET -- cleared on the FIRST real edit
    # (``_mark_library_note_dirty``) or an explicit Save. Drives ONLY
    # the title Input's placeholder-vs-value rendering (see
    # ``LibraryNotesCanvas``'s ``title_placeholder_only``): while this
    # is set for the open note, the title Input shows empty with an
    # "Untitled" placeholder instead of a literal editable "Untitled"
    # value -- the fix for typing landing at the cursor's end and
    # producing e.g. "UntitledAtlas follow-ups". Deliberately NOT used
    # to decide GC-vs-save at exit (see ``session_blank_id``
    # below) -- clearing on the first keystroke is correct for "stop
    # showing the placeholder" but wrong for "should this be GC'd",
    # since a user can type then delete everything back to empty
    # (review round 1, task-2858 T3): that sequence must still GC, so
    # the GC decision needs a flag that survives edits.
    pending_blank_gc_id: str | None = None

    # LIB-14 (review round 1 fix): the id of a note created via "Blank
    # note" THIS SESSION, tracked for the whole session regardless of
    # intermediate edits -- unlike ``pending_blank_gc_id``
    # above, ``_mark_library_note_dirty`` never clears this. Read by
    # ``_flush_library_note_save`` to decide GC-vs-save at exit: when
    # the note being flushed IS this session's blank AND its FINAL live
    # state (title/body/keywords, read fresh, never the stale detail)
    # is empty, the row is GC'd even if it was typed into and emptied
    # out again mid-session (dirty=True) -- covering "type then delete
    # everything" the same as "never touched", which the narrower
    # dirty-gated check above could not. A PRE-EXISTING note the user
    # empties out is never a session blank (this is only ever set by
    # the "Blank note" create path), so it still saves via the normal
    # branch -- the scope guard is structural, not a runtime check.
    # Cleared by: an explicit Save (the user's own "keep it" act,
    # regardless of emptiness -- mirrors ``_library_note_pending_blank_
    # gc_id``'s existing Save-press exemption), or a full editor
    # reset/note switch (``_reset_library_note_editor_state``, the
    # note-row-selection and note_id-deep-link switch sites).
    # Deliberately NOT cleared by autosave persisting non-empty content
    # mid-session: an autosave is not a deliberate "keep this" signal
    # the way an explicit Save press is, so a session blank that got
    # autosaved with real text and was then emptied out again before
    # exit must still GC -- the row is session-created and finally
    # empty, which is exactly the row AC#5 forbids, regardless of what
    # happened to it in between.
    session_blank_id: str | None = None

    # (P0, xhigh review + live-verify round) Whether the user has
    # TOUCHED the title widget during this editor session. The
    # untouched-blank GC used to key blankness on
    # ``raw_title == LIBRARY_NOTE_BLANK_SEED_TITLE`` -- a pure string
    # comparison -- so a note the user deliberately titled "Untitled"
    # (body still empty) was destroyed on navigate-away with no
    # prompt and no undo. A string can never distinguish the seed the
    # create seam wrote from the identical string a human typed; only
    # provenance can, so the GC reads THIS instead. Set on the first
    # title edit (``handle_library_note_title_changed``) and cleared
    # only when a new editor session begins (create / row switch /
    # deep link / full editor reset) -- deliberately NOT cleared by a
    # save, so the distinction survives a save round-trip.
    title_user_edited: bool = False

    # Notes sync panel state. Seeded from config lazily on first entry
    # into sync mode (``_ensure_library_notes_sync_config_loaded``), not
    # here in __init__, so tests/screens that never open the sync panel
    # never pay for a config read.
    sync_config_loaded: bool = False
    sync_direction: str = "bidirectional"
    sync_conflict: str = "newer_wins"
    sync_auto: bool = False
    sync_status: str = "idle"
    sync_activity: tuple[str, ...] = ()
    sync_counter: int = 0
    sync_active_token: int | None = None
    sync_running: bool = False
    auto_sync_timer: Timer | None = None

    # The folder box's live (possibly uncommitted) text. Typing updates
    # only this field -- persisting to the TOML config on every
    # Input.Changed meant a full config rewrite + cache reload per
    # keystroke. It commits to config on Enter, Browse…, or a validated
    # Sync now run. None = not edited this panel visit; fall back to the
    # persisted config value.
    sync_folder_text: str | None = None
