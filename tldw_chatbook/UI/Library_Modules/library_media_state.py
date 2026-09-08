"""``LibraryMediaState`` -- the Media subsystem's own fields.

State PR of the Media extraction series (wave-7 task 1,
``.superpowers/sdd/2026-09-06-library-decomposition-wave7-media``; recipe:
``backlog/docs/library-decomposition-recipe.md``; the prompts series --
``library_prompts_state.py`` -- is the worked example this mirrors most
closely, since Media also has an entangled reader-preferences group and a
bare-underscore third-prefix field). Every field here was moved verbatim out
of ``LibraryScreen.__init__`` (one out of the class body -- see
``arrival_note`` below) in ``tldw_chatbook/UI/Screens/library_screen.py`` --
same default, same type, same comment. The state PR kept every original
``_library_media_<field>``/``_selected_media_id`` attribute name alive as a
generated getter/setter ``@property`` shim on ``LibraryScreen``, between
sentinel comments, so no method body had to be edited; task 3 (the cleanup
PR) retargeted every one of those references to ``self._media_state.<field>``
and DELETED that screen block. The identical generated loop lives on
permanently one layer down, in ``LibraryMediaController``, because the
byte-for-byte canon forbids editing the moved bodies that still spell the
flat names.

**Note on the two ``library_media_state`` modules (basename collision).**
This file (``tldw_chatbook/UI/Library_Modules/library_media_state.py``) is the
SCREEN's own ``__init__``-field state object. The pre-existing
``tldw_chatbook/Library/library_media_state.py`` is a DIFFERENT module -- the
Media domain layer (``MediaBrowseScope``, ``MediaTrashScope``,
``LibraryMediaCanvasState``, ``build_library_media_state``, ...), consumed by
``LibraryScreen._build_library_media_state`` and by the media canvas widgets,
and covered by its own ``Tests/Library/test_library_media_state.py``. This
file imports FROM that one (package-qualified, ``...Library.library_media_
state``); nothing anywhere imports either by bare name, so the shared
basename is inert. The identical pair already exists for prompts
(``library_prompts_state.py`` in both directories) and, less exactly, for six
other subsystems -- it is the established directory precedent, not a new
pattern.

TWO prefixes, not three. The ownership census was run as a SUBSTRING match on
"media" over every ``__init__``-stored attribute (plus a full class-body
``Assign``/``AnnAssign`` scan) rather than a ``startswith`` filter on
``_library_media_``, per the conversations exemplar's own "startswith
enumeration trap" lesson:

- ``_library_media_`` is the DEFAULT prefix -- 81 of the 82 fields. Unlike
  Prompts and Skills there is no plural variant to separate, because "media"
  is already both singular and plural, so no ``MEDIA_PLURAL_STATE_FIELDS``
  constant exists here.
- ``_`` (bare underscore, no "media" word in the prefix position at all)
  covers exactly ONE field: ``selected_media_id`` (the original attribute is
  ``_selected_media_id``) -- the same third-prefix shape the skills series
  found in ``_selected_skill_name`` and the prompts series in
  ``_selected_prompt_id``.

``media_state_shim_attr()`` below is the single function BOTH the screen's
shim-generator loop and this subsystem's wiring test call to resolve a field
name to its original attribute name, rather than each independently
recomputing the branch (the drift risk a duplicated frozenset would
reintroduce).

Field ownership (recipe §2 script, substring "media" over
``__init__``-stored attributes, plus the class-body scan): **85 attributes**
found, of which **82 MOVE**, **2 are WIRING** and **1 is BLOCKED**.

The 2 WIRING attributes are NOT part of this dataclass at all -- exactly the
``_conversation_reader_controller``/``_library_collections_capture_
controller``/``_library_prompt_browse_controller`` "prior-extracted
controller" precedent. Each holds a live controller instance constructed with
lambdas that close over the screen, and each stays a plain ``LibraryScreen``
attribute at its original ``__init__`` position, untouched by this move:

- ``_library_media_browse_controller`` (``LibraryMediaBrowseController``)
- ``_library_media_trash_browse_controller``
  (``LibraryMediaTrashBrowseController``)

The 1 BLOCKED field is ``_library_pending_list_entry_media_return``. Its name
contains "media" and its value is a ``_LibraryMediaReturnReceipt``, but the
FIELD is shared shell state, not Media's: its only two writers are
``_arm_library_list_entry_focus`` and ``_disarm_library_list_entry_focus``,
the general list-entry-focus helpers whose own docstring names their callers
as "a fresh rail-row press landing on Media/Notes/Prompts/Skills, and every
'back to list' exit from that canvas's viewer/editor" -- i.e. four
subsystems, which is the recipe's >=2-subsystems rule by construction (the
prompts controller already binds ``_arm_library_list_entry_focus`` as a named
dependency). It is also one of four members of a single shell family
(``_library_pending_list_entry_focus``,
``_library_pending_list_entry_focus_anchor``,
``_library_list_entry_focus_generation``), assigned and cleared together in
the same two statements; moving one member into a subsystem state object
while its siblings stay on the screen is precisely the split the One Home
Rule exists to prevent. Media's OWN return receipts
(``_library_media_viewer_return``, ``_library_media_trash_return``, the same
type, assigned two lines later) do move.

**Zero fields are BLOCKED by the >=2-subsystems rule itself**, but three
groups need their cross-subsystem readers recorded rather than waved past,
since a name-based tagging heuristic alone would have mis-tagged them (the
recipe's own caveat that the script's tags are name-based, not body-based):

- ``view`` is referenced by 40 methods, 16 of them not media-named. Fifteen
  are shell/plumbing (``compose_content``, ``restore_state``, ``save_state``,
  ``check_action``, ``_library_entry_route_key``, ``_refresh_library_visit_
  surfaces``, ...). The sixteenth, ``_record_library_notes_focus_
  interaction``, is Notes-named and uses a read-only route guard
  (``self._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA and
  self._media_state.view == "list"``) -- it consults where Media is before
  deciding whether a Notes focus interaction applies, exactly as the thirteen
  shell guards do. This matches the prompts series' already-landed precedent
  for ``_library_prompts_mutation_in_flight``, which three Notes-named
  methods likewise read without owning.
- The review-set cluster (``_review_set_picker_worker``,
  ``_review_dismiss_undo_worker``, ``_review_dismiss_receipt_name``,
  ``_walk_active_review_set_unguarded``, ``_active_review_set_banner``,
  ``_active_review_loaded_at_last``, ``_toggle_reviewed_unguarded``,
  ``_review_these_worker``, ``_order_selected_review_pairs``) reads and
  writes ``review_dismiss_receipt`` and reads ``reader_session``. Review sets
  are not one of the eleven subsystems in the recipe's §8 order -- they are a
  media-only feature built ON the media reader (every item is a
  ``backing_media_id``, every walk step calls ``_select_library_media_reader_
  row``), so they tag shell/plumbing and their references move with Media.
- ``reader_layout`` is read by ``_toggle_library_media_reader_pane`` -- the
  generic multi-subsystem pane dispatcher, which is Media-NAMED but reads
  FOUR subsystems' state objects and stays screen-resident (an exclusion for
  this series' controller PR, per the prompts-series ruling).

**The suspend/resume seam (dev's TASK-31521), ruled field by field.** No
screen-lifecycle method reads a moved media field directly except
``_refresh_library_visit_surfaces`` (the ``on_screen_resume`` dispatch seam),
which writes the trash entry trio (``trash_query_draft``,
``trash_input_error``, ``trash_type_choices_visible``) and reads ``view`` --
all shell/plumbing writes that move and are retargeted at cleanup.
``on_screen_suspend`` itself touches NO media field: it calls the two media
METHODS ``_stop_library_media_selection_debounce`` and
``_stop_library_media_filter_timer``, whose bodies are untouched by this move
and keep reaching ``selection_timer``/``filter_timer`` through the shim. The
two Timer fields and the one Worker field (``progress_write_worker``) MOVE,
following the ingest series' ``path_debounce_timer``/``preflight_worker`` and
the prompts series' ``debounce_timer`` -- all three of those are likewise
stopped from ``on_screen_suspend``/``on_unmount``, and both prior series
resolved the same question the same way. The screen-lifecycle fields
themselves (``_library_screen_suspended``, ``_library_visit_entered``) are
NOT media-named, are not in this census, and stay screen-owned.

Four fields keep their *original* ``__init__`` assignment line completely
untouched, because ``self._media_state`` must be constructed EARLY -- right
after ``self._prompts_state`` and BEFORE the shared reader-preferences
tuple-unpack -- for the same reason the collections/skills/prompts series
document, and a field whose original line runs after that forced construction
point cannot be passed as a constructor argument:

- ``reader_preferences`` is one of eight targets unpacked from a single
  ``self._load_library_reader_preference_snapshot()`` call shared with
  Conversations, Notes, File Notes, Collections, Prompts and Skills.
- ``reader_persistence_locks`` depends on a local ``asyncio.Lock()``
  (``library_pane_persistence_lock``) also reused by six other subsystems'
  persistence-lock dicts.
- ``layout_refresh_generation`` mirrors ``self._library_reader_layout_
  refresh_generation``, itself read from the app instance further down.
- ``reader_layout`` is derived from ``reader_preferences`` (via
  ``resolve_media_reader_layout``) after that unpack settles.

Their dataclass defaults below are therefore momentary placeholders,
overwritten by their own original ``__init__`` lines (which route through the
generated shim) before anything else reads them.

Three more fields have genuinely computed (not static-literal) defaults and
become CONSTRUCTOR ARGUMENTS instead, per the recipe's "computed defaults
become constructor arguments so ``__init__`` evaluation order is preserved"
rule and the export series' own ``form`` precedent -- all three read values
that already exist at the early construction point, so their original lines
are deleted outright:

- ``preview_factory`` / ``preview_factory_injected`` read the
  ``preview_widget_factory`` keyword parameter of ``LibraryScreen.__init__``.
- ``analyze_origin`` reads the module constant ``_ANALYZE_ORIGIN_MEDIA``,
  which task 2 relocated to ``Library_Modules/screen_constants.py`` and
  ``library_screen.py`` imports BACK, so it still RESOLVES at the
  ``library_screen`` module path (``Tests/UI/test_library_ingest_
  analyze_skipped.py`` pins that attribute, not the definition site).
  Passing it keeps the
  single-source-of-truth that constant's own comment demands rather than
  re-spelling ``"media"`` here.

Every OTHER field's original line is deleted outright -- 75 in total (74
``__init__`` assignments plus ``arrival_note``'s class-body one): **70** are
static literals or empty collection displays, and **5** are pure
no-argument/one-literal-argument factory calls folded into ``default_factory``
-- ``RowSelection("media")``, ``LibraryMediaReaderSessionState()``,
``MediaBrowseScope()``, ``set()`` and the ``object()`` memo sentinel. Two of
those are frozen dataclasses, one is a plain accumulator whose ``__init__``
only stores its argument and an empty set, and two are builtins; none has a
side effect, so calling them at the earlier construction point is
behaviorally transparent, matching Prompts' own
``PromptSelectionBasket``/``local_prompt_capabilities`` folds.

One moved attribute, ``arrival_note``, was never assigned in
``LibraryScreen.__init__`` at all -- it was a plain class-level annotated
default (``_library_media_arrival_note: str = ""``, task-2223's one-shot
"Matched an existing item" note), read and cleared by the Media-owned
``_pop_library_media_arrival_note`` and set by the shell-owned
``_open_job_in_library``. That is the export series' ``origin_row_id``
precedent exactly: per recipe §2, shell/plumbing-only non-subsystem consumers
still move with the subsystem, the class-level attribute is removed, and this
dataclass's own default (``""``) supplies the identical value.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from textual.timer import Timer
from textual.widget import Widget
from textual.worker import Worker

from ...Library.library_media_reader_state import (
    LibraryMediaReaderSessionState,
    MediaReaderEffectiveLayout,
    MediaReaderLayoutPreferences,
    resolve_media_reader_layout,
)
from ...Library.library_media_state import MediaBrowseScope, MediaTrashScope
from ...Library.row_selection import RowSelection
from ...Widgets.Library.library_media_canvas import LibraryMediaRowScroll
from .screen_support_types import (
    _LibraryMediaReturnReceipt,
    _LibraryMediaReturnSettlement,
    _LibraryMediaSettlementOutcome,
    _LibraryMediaSuccessfulFocusOwnership,
)

#: The single field using the bare ``_`` prefix (no "media" word in the
#: prefix position of the original attribute name). Media has no plural
#: variant, so this is the only exception set -- see the module docstring's
#: two-prefix note.
MEDIA_UNPREFIXED_STATE_FIELDS: frozenset[str] = frozenset({"selected_media_id"})


def media_state_shim_attr(field_name: str) -> str:
    """The original ``LibraryScreen`` attribute name for a state field.

    Single-source resolution of the two-way prefix mapping documented in this
    module's own docstring -- used by BOTH ``LibraryScreen``'s generated shim
    loop and this subsystem's wiring test, so the mapping cannot drift into
    two independently-typed copies.

    Args:
        field_name: A ``LibraryMediaState`` dataclass field name.

    Returns:
        The flat original ``LibraryScreen`` attribute name that field shims
        under.
    """
    if field_name in MEDIA_UNPREFIXED_STATE_FIELDS:
        return "_" + field_name
    return "_library_media_" + field_name


@dataclass
class LibraryMediaState:
    """Every field the Media subsystem exclusively owns."""

    # Placeholder default only -- see the module docstring's
    # forced-early-construction paragraph: the original shared
    # `_load_library_reader_preference_snapshot()` tuple-unpack keeps
    # running, untouched, at its original position.
    reader_preferences: MediaReaderLayoutPreferences = field(
        default_factory=MediaReaderLayoutPreferences
    )

    type_filter: str | None = None
    selection_notice: str = ""
    selected_media_id: str = ""
    select_mode: bool = False
    row_selection: RowSelection = field(
        default_factory=lambda: RowSelection("media")
    )

    # task-2853 AC3: True while the Media Select-mode toolbar's bulk
    # "Delete N selected items?" confirmation should render in place of
    # the normal Select all/Clear/Export selected/Delete selected row.
    confirming_bulk_delete: bool = False

    # task-3020 AC1: set synchronously (before the worker is even
    # scheduled) the instant the confirm row's "Delete" is pressed, and
    # cleared only once ``_delete_library_media_selection`` finishes --
    # a fast double-press on that same button, which stays visible and
    # enabled until the async worker's own completion recompose swaps
    # it away, would otherwise launch a SECOND worker over the same
    # frozen id tuple; ``mark_as_trash`` is idempotent so the delete
    # itself is harmless, but the rail-count decrement is not -- the
    # second worker would decrement it again for ids already gone from
    # ``_local_source_records``. Checked at the very top of the confirm
    # button handler, before it even reads the selection.
    #
    # P1 re-critique finding 3: this flag now ALSO guards
    # ``_undo_library_media_bulk_delete`` -- a single shared flag for
    # both directions, not one flag per direction. Delete and Undo were
    # previously gated by two independent flags AND scheduled into two
    # different exclusive worker groups, so nothing stopped one from
    # starting while the other was still awaiting its own per-item
    # service calls; both mutate the same shared state
    # (``_local_source_records["media"]``, ``_local_source_counts
    # ["media"]``, ``_library_media_delete_receipt_ids``), so an
    # interleaving let Undo finish LAST and clobber a newer delete's
    # writes with its own stale snapshot (or vice versa). Sharing one
    # flag (checked at the top of BOTH button handlers, before either
    # reads any state) makes the two mutually exclusive: whichever
    # press lands first runs to completion in the ``finally`` below
    # before the other can even schedule its worker, and the losing
    # press is a silent no-op rather than a second worker racing the
    # first over the same mutable state.
    #
    # task-14901 (ADR-055): the single-item viewer delete
    # (``handle_library_media_delete_confirm`` /
    # ``_delete_library_media_item``) is the THIRD mutator of that same
    # shared state -- it is one-item bulk, so it claims this same flag
    # and schedules into the same exclusive worker group rather than
    # growing a flag of its own.
    bulk_delete_in_flight: bool = False
    mutation_scope: MediaBrowseScope | None = None
    mutation_authority: int | None = None
    lifecycle_generation: int = 0
    presentation_epoch: int = 0
    current_owner: LibraryMediaRowScroll | None = None
    geometry_floor_owner_identity: int | None = None
    geometry_floor: int = 0
    return_request_id: int = 0
    return_settlement: _LibraryMediaReturnSettlement | None = None
    last_exact_settlement: tuple[_LibraryMediaReturnSettlement, int] | None = None
    last_successful_settlement: (
        tuple[
            _LibraryMediaReturnSettlement,
            int,
            tuple[object, ...],
            tuple[object, ...],
        ]
        | None
    ) = None
    successful_focus_ownership: _LibraryMediaSuccessfulFocusOwnership | None = None
    last_settlement_attempt: tuple[int, int] | None = None
    last_settlement_outcome: (
        tuple[int, _LibraryMediaSettlementOutcome, int | None] | None
    ) = None

    # task-4022 AC2: the ids from the most recently completed media
    # delete (bulk OR, since task-14901, the single-item viewer
    # delete), rendered as a "✓ deleted · N items" receipt (with
    # Undo/Dismiss) until acted on or replaced by a newer delete
    # action. Empty tuple means no receipt to show. Cleared when a new
    # delete confirmation is armed or select mode is freshly
    # entered, set to the succeeded subset when a delete completes, and
    # narrowed to only the still-failed ids by a partial Undo.
    delete_receipt_ids: tuple[str, ...] = ()

    # task-31220: "<n> of <m> · <reason>" while the receipt above
    # names ids whose Undo just FAILED -- the canvas then paints
    # "✗ undo failed · <n> of <m> · <reason>" with "Retry undo"
    # instead of a tick over a recovery that did not happen. Set only
    # by the undo worker; cleared by it on full success and by every
    # path that writes a fresh receipt.
    delete_receipt_undo_failure: str = ""

    # task-31236: (set_id, name, was_active) of the most recently
    # dismissed review set -- rendered as an in-list undo receipt.
    review_dismiss_receipt: tuple[str, str, bool] | None = None

    # task-28007 AC#3/AC#4: the Select-mode bulk-Analyze run. One
    # in-flight flag (a second press is refused with a notice), the
    # receipt's own counts, the failed ids Retry re-runs, and the
    # armed "N already analyzed — Skip them | Overwrite" choice
    # (all_ids, unanalyzed_ids) that AC#3 requires before anything is
    # overwritten. ``_library_media_analyze_reason_cache`` memoises
    # the provider reason for the whole select-mode session: resolving
    # it is not free (Anthropic claude_subscription readiness shells
    # out to the keychain), so it is resolved once per entry, never
    # per sync or per row.
    analyze_running: bool = False
    analyze_total: int = 0
    analyze_done: int = 0
    analyze_failed_ids: tuple[str, ...] = ()
    analyze_choice: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    analyze_reason_cache: str | None = None

    # (fix round 1, I-3) Which surface an in-flight bulk-Analyze run
    # started from -- "media" (Select mode) or "import" (the Import
    # queue's "Analyze N skipped"). Read only by ``on_unmount``'s
    # interrupted-run notice, to send the user back to the control they
    # actually used instead of always naming Select mode's.
    #
    # Placeholder default only -- see the module docstring's
    # constructor-argument paragraph: `__init__` passes
    # `_ANALYZE_ORIGIN_MEDIA`, whose DEFINITION task 2 relocated to
    # `screen_constants.py` (`library_screen` imports it back, so the
    # module-path attribute still resolves) -- one source, cannot drift.
    analyze_origin: str = ""

    # task-4025: "list" | "viewer" | "trash" -- the Trash view is the
    # third in-canvas view of the media canvas (never a rail row or a
    # `type:` cycle value; see the task file's mechanism decision).
    view: str = "list"
    reader_session: LibraryMediaReaderSessionState = field(
        default_factory=LibraryMediaReaderSessionState
    )
    selection_timer: Timer | None = None
    filter_timer: Timer | None = None
    unfiltered_scope: MediaBrowseScope = field(default_factory=MediaBrowseScope)
    unfiltered_selected_id: str = ""
    filter_restore_id: str = ""
    filter_select_first: bool = False

    # Placeholder defaults only -- the persistence-locks dict literal (which
    # shares the local `library_pane_persistence_lock`), the
    # layout-refresh-generation mirror and the `resolve_media_reader_layout`
    # call all keep their ORIGINAL `__init__` positions (see the module
    # docstring's forced-early-construction paragraph), writing into this
    # object's own fields the instant each original line runs.
    reader_persistence_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    layout_refresh_generation: int = 0
    reader_layout: MediaReaderEffectiveLayout = field(
        default_factory=lambda: resolve_media_reader_layout(
            0,
            MediaReaderLayoutPreferences(),
        )
    )

    # task-14902: True while the media type chooser's direct-pick strip
    # replaces the browse toolbar row (the Notes Sort strip pattern).
    type_choices_visible: bool = False

    # task-28013: the browse sort chooser's direct-pick strip visibility.
    sort_choices_visible: bool = False

    # Trash owns an independent source scope. Draft and semantic focus
    # remain screen concerns because Task 5 renders their controls; page
    # authority lives exclusively in ``_library_media_trash_browse_controller``.
    trash_query_draft: str = ""
    trash_input_error: str = ""
    trash_type_choices_visible: bool = False
    trash_focus_identity: str = "#library-media-trash-row-0"
    trash_focus_authority_generation: int = 0
    trash_focus_request_key: tuple[MediaTrashScope, str] | None = None
    trash_mounted_authority: bool = False
    detail: Mapping[str, Any] | None = None

    # task-15458: the exact detail object the last viewer compose rendered,
    # compared by IDENTITY. ``_refresh_library_media_detail``'s arrival
    # recompose is skipped when it matches, which is what keeps a long
    # document from being parsed twice per open (see
    # ``_recompose_library_media_detail_if_unrendered``). Reset to None
    # wherever ``_library_media_detail`` is cleared, so a service that
    # hands back a cached Mapping cannot make a fresh open look "already
    # rendered" and strand the viewer on its loading line.
    composed_detail: Mapping[str, Any] | None = None
    editing: bool = False
    confirming_delete: bool = False
    highlights: list[dict[str, Any]] = field(default_factory=list)
    editing_analysis: bool = False

    # task-28006: an LLM analysis generation is in flight for the open item.
    generating_analysis: bool = False
    content_query: str = ""
    content_match_index: int = 0

    # task-31237: the content Find bar is collapsed until the Find
    # action opens it -- a permanently open "Search content…" input
    # duplicated the Find button and spent 3 rows on every fresh item.
    find_open: bool = False

    # task-31269: one-shot Find-gesture token; spent by the next viewer
    # build/sync so an item change can never move focus into the
    # search Input.
    find_focus_pending: bool = False

    # task-22209: in-content match list for the open item, memoized on
    # (detail object identity, query). Both the query submit and every
    # Prev/Next click need it, and deriving it costs a full content
    # copy (``build_library_media_viewer_state``) plus a full scan --
    # per click, on a document that has not changed. The detail is only
    # ever replaced wholesale (a fetch settles) or cleared to None,
    # never mutated in place, so its identity is a sound document
    # marker; the None sentinel guarantees the first lookup misses.
    # task-28026: keyed by (detail, query, MODE) -- the Read and Analysis
    # tabs search different corpora of the same detail, so the mode is
    # part of the key or a tab switch would serve the other tab's matches.
    content_match_memo: tuple[Any, str, tuple[int, ...], str] | None = None

    # LIB-13: "rendered" (Markdown, via the same render path Notes
    # Preview uses) or "raw" (plain/highlighted text). Reseeded per
    # item by ``_refresh_library_media_detail`` from the freshly built
    # viewer state's ``is_markdown`` (rendered default for markdown
    # media, raw for everything else); the toggle handlers below flip
    # it in place without touching the default-selection logic.
    content_mode: str = "raw"
    read_scroll_by_id: dict[str, tuple[int, int]] = field(default_factory=dict)
    progress_restored_id: str | None = None

    # TASK-22210: reading-progress writes are coalesced to the latest
    # per-item value and drained by one serial worker (mirrors the
    # lifecycle-persistence pattern above; cancellation-based supersede
    # is unsound for durable writes -- see task-1541's lesson).
    progress_pending_writes: dict[str, tuple[int | str, tuple[int, int]]] = field(
        default_factory=dict
    )
    progress_inflight_write: tuple[str, int | str, tuple[int, int]] | None = None
    progress_persisted_offsets: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )
    progress_write_worker: Worker | None = None

    # Task 21665: decoded local originals are ephemeral screen-session
    # state. The production renderer is imported only after the capability
    # gate passes, preserving Library's no-Pillow startup path.
    #
    # Placeholder defaults only -- see the module docstring's
    # constructor-argument paragraph: `__init__` passes both from its own
    # `preview_widget_factory` keyword parameter.
    preview_factory: Callable[..., Widget] | None = None
    preview_factory_injected: bool = False
    preview_images: dict[str, Any] = field(default_factory=dict)
    preview_status: dict[str, str] = field(default_factory=dict)
    preview_hidden: set[str] = field(default_factory=set)
    preview_loading: dict[str, int] = field(default_factory=dict)

    # task-22208: viewer display-state memo, keyed by the DETAIL OBJECT
    # (identity) plus the build parameters. ``build_library_media_
    # viewer_state`` copies the whole content string per call
    # (``str(content).strip()``), so it must run once per detail
    # ARRIVAL, not once per sync. The detail is only ever replaced
    # wholesale (worker settle) or cleared to None -- never mutated in
    # place -- so identity is a sound arrival marker; the sentinel
    # guarantees the first call always misses. See
    # ``_library_media_viewer_state_cached`` for the full key.
    viewer_state_memo_detail: Any = field(default_factory=object)
    viewer_state_memo_states: dict[tuple[str, str, str, bool], Any] = field(
        default_factory=dict
    )
    viewer_return: _LibraryMediaReturnReceipt | None = None
    trash_return: _LibraryMediaReturnReceipt | None = None

    #: One-shot note the media viewer surfaces on its next build --
    #: set when navigation arrives via a dedup-matched ingest row
    #: (task-2223: "Open in Library" on a match landed on the twin's
    #: identity with no explanation).
    arrival_note: str = ""
