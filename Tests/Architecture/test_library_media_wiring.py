"""Media extraction series: state object exists and is screen-wired.

Wave-7 Task 1 (media series 1/3, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; prompts series precedent: ``Tests/
Architecture/test_library_prompts_wiring.py``, its state-PR-era shape --
the closest match, since Media likewise carries a bare-underscore
third-prefix field and an entangled reader-preferences group). Every field
``LibraryMediaState`` declares must have a matching generated property shim
-- on ``LibraryScreen`` while the state PR's block existed, and, since task
3 deleted that block, on ``LibraryMediaController`` alone -- resolved via
``media_state_shim_attr()``, the
single-source two-way prefix mapping (``_library_media_`` default, bare
``_`` for the one unprefixed field, ``selected_media_id``) documented in
``library_media_state.py``'s own module docstring. Unlike a looser "either
prefix works" check, these tests assert the EXACT expected shim name per
field, and that each shim genuinely reads AND writes the state object rather
than merely existing as a property.

**What the per-field sweep can and cannot prove.** Because the shim
loop and this file's sweep both call ``media_state_shim_attr()``, the sweep
proves the two are CONSISTENT -- it cannot prove the mapping is CORRECT. The
prompts series proved that hole is real: deleting a prefix branch from the
shared resolver outright left its sweep passing, because generator and test
then agreed on the same wrong answer. ``test_media_state_shim_attr_maps_
each_prefix_family_to_its_literal_name`` below closes it with hard-coded
expected strings -- one per prefix family, including the bare-underscore one
-- and is carried here from BIRTH rather than added by a later review round.

The state module under test is ``tldw_chatbook.UI.Library_Modules.library_
media_state`` -- NOT the unrelated, pre-existing ``tldw_chatbook.Library.
library_media_state`` domain module of the same basename (see that
docstring's own note on the collision).

**Task 2 (controller PR)** adds the full-cluster ownership / same-name-
delegator-forwarding / staticmethod-class-forwarding / controller-state-shim
checks (``_MEDIA_CLUSTER_METHOD_NAMES``, 142 names) plus a
constructor-binding coverage check (``_MEDIA_CONTROLLER_BOUND_NAMES``, 92
names) carried from the prompts series, which added it because the skills
series shipped a silent production regression precisely in that gap (a moved
body's ``getattr(self, "focused", None)`` with no ``focused`` property
bound; recipe §3's unbound-attribute-escape entry). Media's own instance of
that shape is one indirection further out and is called out on the tuple
itself. See ``library_media_controller.py``'s own module docstring for the
full 251-candidate derivation, the 111 exclusions, and the
connected-components evidence behind the single-controller decision.

**Task 3 (cleanup PR)** deleted the screen's shim block, so
``test_state_object_fields_match_the_shim_surface`` narrowed to
``test_state_object_declares_the_censused_field_count`` (the half that never
depended on the screen), ``test_every_shim_reads_and_writes_its_own_state_
field`` was re-aimed at the controller's own permanent shim loop rather than
deleted, ``test_the_screen_no_longer_carries_a_media_state_shim`` was added
asserting ABSENCE, and ``_MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED`` was filled
with the 22 zero-reference delegator names it deleted.

**Phase C, task 3 (region ownership)** removed a further 16 delegators for a
different reason -- not deadness but OWNERSHIP: their messages originate
inside ``LibraryMediaCanvas``, so the region widget catches them directly
now. They live in their own set,
``_MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS``, because merging them into
the zero-reference set would make that set's stated criterion false.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from tldw_chatbook.UI.Library_Modules.library_media_state import (
    MEDIA_UNPREFIXED_STATE_FIELDS,
    LibraryMediaState,
    media_state_shim_attr,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEDIA_CONTROLLER_PATH = (
    _REPO_ROOT / "tldw_chatbook/UI/Library_Modules/library_media_controller.py"
)
_LIBRARY_SCREEN_PATH = _REPO_ROOT / "tldw_chatbook/UI/Screens/library_screen.py"

#: The ownership census's own MOVE count (recipe §2 script, substring
#: "media" over every ``__init__``-stored attribute of ``LibraryScreen``,
#: plus a full class-body ``Assign``/``AnnAssign`` scan that found the one
#: class-level-only attribute, ``_library_media_arrival_note``): 85
#: attributes found, 2 WIRING (live controller instances -- see below), 1
#: BLOCKED (``_library_pending_list_entry_media_return``, shared shell
#: state -- see the state module's docstring), so 82 move. Pinned here so a
#: field silently added to or dropped from the dataclass fails loudly
#: instead of quietly resizing the shim surface this file checks.
#: 2026-10-19, TASK-34000.24: 82 -> 83, ``link_receipt`` (the Reader's
#: "Use in Console" workspace-link receipt; shim ``_library_media_link_receipt``).
#: 2026-10-20, TASK-34000.18 fix round 2: 83 -> 84, ``highlight_held_by_reader``
#: (the order-based hold: the Reader holds the list highlight from an open or
#: select until the user's next scope change), placed beside ``editing``.
#: 2026-10-20, TASK-34000.22: 84 -> 86, ``edit_draft`` beside ``editing`` and
#: ``analysis_draft`` beside ``editing_analysis`` (the two authoring forms'
#: typed text; shims ``_library_media_edit_draft`` /
#: ``_library_media_analysis_draft``).
#: 2026-11-15, TASK-36952.3.6: 86 -> 88, ``rename_cache`` (was the screen's
#: ``_library_media_rename_cache``) and ``review_done_map_cache`` (was the
#: screen's lazily created ``_review_done_map_cache``; bare-underscore shim),
#: each read only by a body that moved to the controller.
#: 2026-10-27, TASK-34000.40 fix round 1 (review I-3): 86 -> 87,
#: ``saving_analysis`` beside ``analysis_draft`` (the analysis form's own
#: Save in flight, so a double or held Ctrl+S writes one version; shim
#: ``_library_media_saving_analysis``). Rebased 2026-11-19 onto
#: TASK-36952.3.6's 88: 88 -> 89.
_EXPECTED_MEDIA_STATE_FIELD_COUNT = 89

#: The 2 WIRING attributes the state PR deliberately left on
#: ``LibraryScreen`` (the ``_conversation_reader_controller``/``_library_
#: collections_capture_controller``/``_library_prompt_browse_controller``
#: precedent): each holds a live prior-extracted controller instance, not
#: data.
_MEDIA_WIRING_SCREEN_ATTRS: tuple[str, ...] = (
    "_library_media_browse_controller",
    "_library_media_trash_browse_controller",
)

#: The 1 BLOCKED attribute. Its name contains "media" and it holds a
#: ``_LibraryMediaReturnReceipt``, but the FIELD is shared shell state: its
#: only writers are the general ``_arm_library_list_entry_focus``/
#: ``_disarm_library_list_entry_focus`` helpers, whose callers span
#: Media/Notes/Prompts/Skills (the recipe's >=2-subsystems rule), and it is
#: one of four members of a shell family assigned and cleared together.
_MEDIA_BLOCKED_SCREEN_ATTR = "_library_pending_list_entry_media_return"


@pytest.mark.unit
def test_state_object_declares_the_censused_field_count() -> None:
    """The ownership census's MOVE count, pinned against the dataclass.

    Task 1 paired this with a per-field screen-shim sweep; task 3 deleted
    that shim block (every screen-side reference now spells
    `self._media_state.<field>` outright), so what survives here is the half
    that never depended on the screen: a field silently added to or dropped
    from `LibraryMediaState` must fail loudly rather than quietly resizing
    every surface this file checks.
    """
    field_names = {f.name for f in dataclasses.fields(LibraryMediaState)}
    assert field_names, "state object is empty"
    assert len(field_names) == _EXPECTED_MEDIA_STATE_FIELD_COUNT, (
        f"expected {_EXPECTED_MEDIA_STATE_FIELD_COUNT} media fields, "
        f"got {len(field_names)}"
    )


@pytest.mark.unit
def test_every_controller_shim_reads_and_writes_its_own_state_field() -> None:
    """Each generated property is a real two-way shim, not a stub.

    A getter/setter pair that existed but bound the WRONG field (the
    closure-binding trap a `for` loop over `dataclasses.fields` invites --
    every generated property capturing the LAST field unless the name is
    bound as a default argument) would satisfy a bare `isinstance(...,
    property)` check while silently aliasing 82 names onto one field. This
    round-trips a distinct sentinel through every name to rule that out.

    Task 1 aimed this at the SCREEN's shim block; task 3 deleted that block
    and re-aimed the same probe at `LibraryMediaController`'s own permanent
    shim loop, which is generated by the identical `dataclasses.fields` loop
    and so carries the identical trap (the prompts series' own deviation
    from the earlier delete-it precedent, taken for the same reason: the
    surviving `test_media_controller_exposes_every_state_field` below is
    exactly the weaker `isinstance(..., property)` check this one exists to
    strengthen).
    """
    from tldw_chatbook.UI.Library_Modules.library_media_controller import (
        LibraryMediaController,
    )

    controller = object.__new__(LibraryMediaController)
    state = LibraryMediaState()
    controller._media_state_accessor = lambda: state

    field_names = sorted(f.name for f in dataclasses.fields(LibraryMediaState))
    read_mismatch = []
    write_mismatch = []
    for name in field_names:
        shim_attr = media_state_shim_attr(name)
        if getattr(controller, shim_attr) is not getattr(state, name):
            read_mismatch.append(shim_attr)
        sentinel = object()
        setattr(controller, shim_attr, sentinel)
        if getattr(state, name) is not sentinel:
            write_mismatch.append(shim_attr)
        if getattr(controller, shim_attr) is not sentinel:
            read_mismatch.append(shim_attr)
    assert not read_mismatch, f"shim getters do not read their field: {read_mismatch!r}"
    assert not write_mismatch, (
        f"shim setters do not write their field: {write_mismatch!r}"
    )

    # Every field ended up holding a DISTINCT sentinel -- proof no two shims
    # share one underlying field.
    written = [getattr(state, name) for name in field_names]
    assert len({id(value) for value in written}) == len(field_names)


@pytest.mark.unit
def test_the_screen_no_longer_carries_a_media_state_shim() -> None:
    """The task-1 screen shim block is GONE, and must stay gone.

    Mirrors every prior series' own post-cleanup assertion: the flat
    `_library_media_<field>`/`_selected_media_id` names must not exist on
    `LibraryScreen` in ANY form -- neither as the deleted generated
    properties nor as a hand-added replacement -- so a reintroduced shim
    fails loudly here instead of silently re-splitting the ownership this
    series consolidated.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    still_shimmed = [
        media_state_shim_attr(f.name)
        for f in dataclasses.fields(LibraryMediaState)
        if hasattr(LibraryScreen, media_state_shim_attr(f.name))
    ]
    assert not still_shimmed, (
        f"the media-state shim block is back on LibraryScreen: {still_shimmed!r}"
    )


@pytest.mark.unit
def test_media_state_shim_attr_maps_each_prefix_family_to_its_literal_name() -> None:
    """The two-way mapping, pinned against LITERAL expected strings.

    The only assertion in this file that is not self-referential: every
    other check resolves the expected name by calling the same
    `media_state_shim_attr()` the screen's own shim loop calls, so screen
    and test agree even when the mapping is wrong. The prompts series
    proved that hole empirically (deleting its plural branch left the rest
    of its wiring file green), which is why this one ships from birth. One
    field per prefix family, spelled out:
    """
    # default family -- `_library_media_` (81 of the 82 fields)
    assert media_state_shim_attr("view") == "_library_media_view"
    assert media_state_shim_attr("arrival_note") == "_library_media_arrival_note"
    # bare-underscore family -- the one field with no "media" prefix word
    assert media_state_shim_attr("selected_media_id") == "_selected_media_id"
    # TASK-36952.3.6: a moved body reads this one as `getattr(self,
    # "_review_done_map_cache", None)`, so a wrong mapping would default silently
    assert media_state_shim_attr("review_done_map_cache") == "_review_done_map_cache"


@pytest.mark.unit
def test_media_state_field_prefix_set_is_real_state_fields() -> None:
    """Guards the single-source prefix mapping against drift: every name the
    exception set lists must actually be a `LibraryMediaState` field (a typo
    or a stale entry here would otherwise silently shim nothing under the
    intended prefix and everything under the wrong one instead).
    """
    field_names = {f.name for f in dataclasses.fields(LibraryMediaState)}
    unknown_bare = MEDIA_UNPREFIXED_STATE_FIELDS - field_names
    assert not unknown_bare, (
        f"MEDIA_UNPREFIXED_STATE_FIELDS names unknown fields: {unknown_bare!r}"
    )


@pytest.mark.unit
def test_wiring_and_blocked_attributes_stay_off_the_state_object() -> None:
    """The 2 media controller attributes are WIRING and the 1 shared
    list-entry receipt is BLOCKED shell state -- none may be a
    `LibraryMediaState` field, and all three must remain plain
    (non-property) attributes on `LibraryScreen`.
    """
    field_names = {f.name for f in dataclasses.fields(LibraryMediaState)}
    assert "browse_controller" not in field_names
    assert "trash_browse_controller" not in field_names
    assert "pending_list_entry_media_return" not in field_names

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    shimmed = [
        attr
        for attr in (*_MEDIA_WIRING_SCREEN_ATTRS, _MEDIA_BLOCKED_SCREEN_ATTR)
        if isinstance(getattr(LibraryScreen, attr, None), property)
    ]
    assert not shimmed, f"wiring/blocked attributes were shimmed as state: {shimmed!r}"


#: Every method Task 2 moved into ``LibraryMediaController``, under its
#: original ``LibraryScreen`` name. Derived from a full ``ast`` census of
#: every ``LibraryScreen`` class-body method whose name contains "media"
#: (case-insensitive): **251 raw ``FunctionDef`` matches, 251 unique names**
#: (no property/setter-pair gap) -- minus 111 exclusions: 73 unbound-fake-
#: self, 16 instance-attribute-monkeypatch, 8 source-census, 4
#: module-globals-coupling, 3 screen-identity (``self in <widget>.ancestors``
#: -- recipe §3's sixth bypass shape in a MEMBERSHIP form its own ``is``/``is
#: not`` census cannot see), 2 class-monkeypatch, 2 bypassed-construction-
#: lifecycle, 1 callback-identity, 1 shared-shell-helper
#: (``_sanitize_media_field``, which the Prompts and Notes clusters also
#: call) and 1 generic dispatcher (``_toggle_library_media_reader_pane``).
#: The last THREE classes -- 6 names -- were found by this task's own
#: BATTERY, not by its static census: see the controller docstring's
#: exclusion classes 8, 9 and 10, and the amended-RED-tuple rule in recipe §3
#: that makes such a correction the expected shape of the work rather than a
#: smell.
#: NOT a prefix/substring shortcut. See ``library_media_controller.py``'s
#: module docstring for the full per-name reasoning behind every exclusion,
#: and for the connected-components evidence behind the single-controller
#: decision.
_MEDIA_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_add_library_media_highlight",
    "_adopt_library_media_highlight",
    "_adopt_library_media_row_owner",
    "_advance_library_media_content_match",
    "_advance_library_media_presentation_epoch",
    "_apply_library_media_active_surface",
    "_apply_library_media_list_return",
    "_bind_library_media_settlement_deadline",
    "_bounded_library_media_trash_title",
    "_build_library_media_active_child",
    "_build_library_media_reader",
    "_build_library_media_state",
    "_build_library_media_viewer_display_state",
    "_cache_library_media_preview",
    "_cancel_library_media_selection_settlement",
    "_cancel_library_media_trash_delete_confirmation",
    "_capture_library_media_focus_identity",
    "_capture_library_media_trash_return",
    "_consume_library_media_find_focus",
    "_delete_library_media_highlight",
    "_disarm_library_media_return_for_route_change",
    "_dispatch_library_media_detail_request",
    "_expire_library_media_return_settlement",
    "_fetch_library_media_analysis_detail",
    "_fetch_library_media_highlights",
    "_finish_library_media_list_return",
    "_focus_library_media_content_search_input",
    "_focus_library_media_items_pane",
    "_focus_library_media_page_control",
    "_focus_library_media_trash_after_paint",
    "_focus_library_media_trash_entry",
    "_focus_library_media_trash_intent",
    "_handle_library_media_row_geometry_changed",
    "_library_media_adjacent_row",
    "_library_media_analyze_reason",
    "_library_media_analyze_receipt_fields",
    "_library_media_canvas_presentation",
    "_library_media_console_representation",
    "_library_media_exact_return_candidate",
    "_library_media_image_preview_capable",
    "_library_media_image_preview_projection",
    "_library_media_item_traversal_active",
    "_library_media_list_surface_active",
    "_library_media_live_focus_is_allowed",
    "_library_media_preview_request_is_current",
    "_library_media_reader_identity",
    "_library_media_reader_target_id",
    "_library_media_editor_dirty",
    "_notify_media_editor_dirty_veto",
    "_reveal_library_media_dirty_form",
    "_library_media_request_matches_current_authority",
    "_library_media_semantic_row_is_current",
    "_library_media_successful_focus_is_allowed",
    "_library_media_trash_applied_scope",
    "_library_media_trash_canvas_presentation",
    "_library_media_viewer_substate_active",
    "_load_library_media_image_preview",
    "_load_library_media_list_if_needed",
    "_mirror_library_media_reader_preference",
    "_mounted_library_media_viewer",
    "_navigate_to_media",
    "_notify_library_media_highlight_warning",
    "_notify_library_media_read_later_warning",
    "_open_library_external_media_detail",
    "_open_library_media_viewer",
    "_patch_local_media_record",
    "_pop_library_media_arrival_note",
    "_project_library_media_stage_classes",
    "_recompose_library_media_detail_if_unrendered",
    "_reload_library_media_highlights",
    "_request_library_media_facets",
    "_request_library_media_filter",
    "_request_library_media_page",
    "_request_library_media_sort",
    "_request_library_media_trash_page",
    "_resize_library_media_reader_shell",
    "_resolve_library_media_trash_focus_target",
    "_restore_library_media_focus",
    "_restore_library_media_scope",
    "_retry_library_media_browse",
    "_schedule_library_media_image_preview",
    "_scroll_library_media_content_to_line",
    "_select_library_media_adjacent_item",
    "_select_library_media_reader_row",
    "_selected_media_handoff_payload",
    "_set_library_media_content_mode",
    "_settle_library_media_return_from_geometry",
    "_start_library_media_read_later_toggle",
    "_sync_library_media_viewer_mutation_gate",
    "_sync_library_media_viewer_or_recompose",
    "_sync_library_media_viewer_state",
    "_toggle_library_media_read_later",
    "_valid_library_media_trash_delete_ack",
    "action_library_media_bulk_delete_cancel",
    "action_library_media_next_item",
    "action_library_media_prev_item",
    "action_library_media_toggle_select_mode",
    "action_library_media_trash_back",
    "handle_library_media_analysis_cancel",
    "handle_library_media_analysis_edit",
    "handle_library_media_analysis_save",
    "handle_library_media_analysis_text_changed",
    "handle_library_media_edit_field_changed",
    "handle_library_media_back",
    "handle_library_media_content_mode_raw",
    "handle_library_media_content_mode_rendered",
    "handle_library_media_content_search_next",
    "handle_library_media_content_search_prev",
    "handle_library_media_content_search_submitted",
    "handle_library_media_delete_cancel",
    "handle_library_media_edit",
    "handle_library_media_edit_cancel",
    "handle_library_media_export",
    "handle_library_media_filter_changed",
    "handle_library_media_filter_clear",
    "handle_library_media_filter_submitted",
    "handle_library_media_highlight_add",
    "handle_library_media_highlight_delete",
    "handle_library_media_image_preview_retry",
    "handle_library_media_image_preview_toggle",
    "handle_library_media_next",
    "handle_library_media_open",
    "handle_library_media_open_original",
    "handle_library_media_open_viewer",
    "handle_library_media_previous",
    "handle_library_media_read_later",
    "handle_library_media_reader_mode",
    "handle_library_media_reader_more",
    "handle_library_media_reader_retry",
    "handle_library_media_retry",
    "handle_library_media_review_selected",
    "handle_library_media_review_sets",
    "handle_library_media_review_these",
    "handle_library_media_select_all",
    "handle_library_media_select_clear",
    "handle_library_media_sort",
    "handle_library_media_sort_choice",
    "handle_library_media_trash_delete_cancel",
    "handle_library_media_trash_next",
    "handle_library_media_trash_previous",
    "handle_library_media_trash_retry",
    "handle_library_media_trash_search_changed",
    "handle_library_media_trash_search_submitted",
    "handle_library_media_trash_type_choice",
    "handle_library_media_trash_type_filter",
    "request_library_media_layout_refresh",
    "use_media_in_chat",
    # TASK-36952.3.1 (2026-11-15): the Reader's Find, the Escape label and
    # viewer-back ladder, and the Console handoff -- 19 bodies.
    "_after_library_media_viewer_sync",
    "_close_library_media_find",
    "_exit_library_media_viewer",
    "_library_media_escape_label",
    "_library_media_find_state",
    "_library_media_find_unavailable_reason",
    "_library_media_reader_exit_available",
    "_open_selected_media_handoff",
    "_queue_after_library_media_viewer_recompose",
    "_reset_library_media_search_on_mode_change",
    "_set_library_media_link_receipt",
    "_toggle_library_media_find",
    "action_library_media_move_to_trash",
    "action_library_media_read_later",
    "action_library_media_reader_find",
    "action_library_media_use_in_console",
    "action_library_media_viewer_back",
    "handle_library_media_link_undo",
    "handle_library_media_reader_find",
    # TASK-36952.3.2 (2026-11-15): the ADR-055 mutation interlock, bulk and
    # single-item delete, Undo, the selection export and the edit save -- 24
    # bodies.
    "_begin_library_media_mutation",
    "_cancel_library_media_bulk_delete",
    "_claim_library_media_mutation",
    "_complete_library_media_mutation",
    "_delete_library_media_item",
    "_delete_library_media_selection",
    "_library_media_backing_id",
    "_library_media_mutation_summary",
    "_library_media_row_walk_leaves_dirty_form",
    "_notify_library_media_delete_warning",
    "_notify_library_media_edit_warning",
    "_required_library_media_backing_id",
    "_reset_library_media_reader_to_no_selection",
    "_save_library_media_edit",
    "_undo_library_media_bulk_delete",
    "handle_library_media_bulk_delete_cancel",
    "handle_library_media_bulk_delete_confirm",
    "handle_library_media_bulk_delete_receipt_dismiss",
    "handle_library_media_bulk_delete_undo",
    "handle_library_media_delete",
    "handle_library_media_delete_confirm",
    "handle_library_media_delete_selected",
    "handle_library_media_edit_save",
    "handle_library_media_export_selected",
    # TASK-36952.3.3 (2026-11-15): select mode, the browse request and sync,
    # the type filter and the take-note action -- 23 bodies.
    "_clear_library_media_selection_for_scope_change",
    "_exit_library_media_select_mode",
    "_library_media_list_unselectable",
    "_library_media_select_enter_available",
    "_library_media_type_options",
    "_notify_library_media_selection_discarded",
    "_request_library_media_browse",
    "_request_library_media_type",
    "_start_library_note_from_media_source",
    "_stop_library_media_filter_timer",
    "_stop_library_media_selection_debounce",
    "_sync_library_media_browse_state",
    "_take_library_media_note",
    "_toggle_library_media_select_mode",
    "action_library_media_take_note",
    "action_library_media_toggle_row_selection",
    "handle_library_media_empty_clear_type",
    "handle_library_media_empty_import",
    "handle_library_media_row",
    "handle_library_media_select_toggle",
    "handle_library_media_take_note",
    "handle_library_media_type_choice",
    "handle_library_media_type_filter_pressed",
    # TASK-36952.3.4 (2026-11-15): the Trash view's state build, sync, focus,
    # Back, restore and permanent delete -- 19 bodies.
    "_build_library_media_trash_state",
    "_exit_library_media_trash",
    "_library_media_trash_action_disabled_reason",
    "_library_media_trash_actions_live",
    "_library_media_trash_focus_selectors",
    "_library_media_trash_retry_visible",
    "_permanently_delete_library_media_from_trash",
    "_press_library_media_trash_action",
    "_restore_library_media_from_trash",
    "_sync_library_media_trash_state",
    "_validated_library_media_trash_restore_summary",
    "action_library_media_trash_delete",
    "action_library_media_trash_restore",
    "handle_library_media_trash_back",
    "handle_library_media_trash_delete",
    "handle_library_media_trash_delete_confirm",
    "handle_library_media_trash_open",
    "handle_library_media_trash_restore",
    "handle_library_media_trash_row",
    # TASK-36952.3.5 (2026-11-15): the Reader's analysis provider gate, Generate,
    # dispatch and save, and the bulk Analyze run with its receipt actions --
    # 17 bodies.
    "_analyze_library_media_selection",
    "_analyze_one_library_media_item",
    "_clear_library_media_analyze_receipt",
    "_dispatch_library_media_analysis",
    "_generate_library_media_analysis",
    "_library_media_analysis_provider_reason",
    "_library_media_unanalyzed_ids",
    "_notify_library_media_analysis_warning",
    "_reproject_library_media_analysis_row",
    "_save_library_media_analysis",
    "_start_library_media_analyze",
    "handle_library_media_analysis_generate",
    "handle_library_media_analyze_overwrite",
    "handle_library_media_analyze_receipt_dismiss",
    "handle_library_media_analyze_retry",
    "handle_library_media_analyze_selected",
    "handle_library_media_analyze_skip",
    # TASK-36952.3.6 (2026-11-15): the Reader's detail load, reading-progress
    # fetch, capture, write queue and restore, the memoised viewer state and
    # match finder, the speaker-rename facts and handler, and the browse rows'
    # review decoration -- 16 bodies.
    "_cache_library_media_reading_progress",
    "_capture_library_media_loaded_progress",
    "_decorate_library_media_reviewed",
    "_drain_library_media_progress_writes",
    "_fetch_library_media_reading_progress",
    "_handle_library_media_speaker_renamed",
    "_library_media_can_rename_speakers",
    "_library_media_content_matches",
    "_library_media_progress_write_is_current",
    "_library_media_selected_backing_id",
    "_library_media_speaker_rename_facts",
    "_library_media_viewer_state_cached",
    "_queue_library_media_progress_write",
    "_refresh_library_media_detail",
    "_restore_library_media_loaded_progress",
    "_write_library_media_loaded_progress",
    # TASK-36952.3.7 (2026-11-17): the return-settlement machinery, the
    # adaptive reader layout sync and its grip focus and open-time width
    # restore, the "‹ Library" rail return, the empty-list focus fallback, the
    # rail-switch Reader keep, the surfaces-or-recompose seam, and the
    # review-set exit, toggle, Undo and receipt-close actions -- 23 bodies.
    "_arm_library_media_return_settlement",
    "_build_library_media_rail_return",
    "_commit_library_media_return",
    "_focus_library_media_grip_if_current",
    "_library_media_content_signature",
    "_library_media_empty_list_fallback_target",
    "_library_media_focus_target_matches_receipt",
    "_library_media_layout_signature",
    "_library_media_rail_return_visible",
    "_library_media_return_candidate",
    "_library_media_settlement_tree",
    "_rail_switch_keeps_media_reader",
    "_reconcile_library_media_stage_presentation",
    "_resolve_library_media_settlement_target",
    "_restore_library_media_reader_width_on_open",
    "_sync_library_media_rail_return",
    "_sync_library_media_reader_layout_from_shell",
    "_sync_library_media_surfaces_or_recompose",
    "action_library_media_exit_review",
    "action_library_media_toggle_reviewed",
    "handle_library_media_rail_return",
    "handle_library_media_review_dismiss_receipt_close",
    "handle_library_media_review_dismiss_undo",
    # TASK-36952.3.8 (2026-11-17): ruling R3 -- the shared field sanitizer the
    # Prompts, Skills and Notes clusters also call; its screen name stays as a
    # static delegator, so their bindings are unchanged.
    "_sanitize_media_field",
)

#: The names above that are ``@staticmethod`` on ``LibraryScreen``. Their
#: delegators forward straight to the module-level ``LibraryMediaController``
#: CLASS (the conversations exemplar's "static-method delegator pattern",
#: reused by every series since), not through ``self._media_controller``.
_MEDIA_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_bounded_library_media_trash_title",
        "_restore_library_media_scope",
        "_valid_library_media_trash_delete_ack",
        # TASK-36952.3.4: moved with the Trash restore; its delegator is pruned.
        "_validated_library_media_trash_restore_summary",
        # TASK-36952.3.8: ruling R3's shared sanitizer.
        "_sanitize_media_field",
    }
)

#: The moved names whose screen delegator nothing reaches through a screen
#: or ``LibraryScreen``. Every remaining reference to such a name is the
#: controller's own ``self.<name>``, a ``*_media_controller`` /
#: ``LibraryMediaController`` receiver (a ``monkeypatch.setattr`` or
#: ``getattr`` string on one, and a substring check against a controller
#: method's source, count as such), a fake's own attribute (a
#: ``SimpleNamespace(<name>=...)`` keyword, an attribute assigned on a fake,
#: or a ``def`` on a fake class), a pin in this file, or prose. Census roots:
#: ``tldw_chatbook/``, ``Tests/``, ``scripts/``, ``Helper_Scripts/`` and
#: ``Docs/``. TASK-36952.3.1-3.8 pruned by this rule.
#:
#: History: wave-7 task 3 (the media series' own cleanup, 3/3) applied the
#: stricter rule of ZERO references outside the delegator's own body, across
#: all four census spellings (attribute, quoted-string, bare-assignment,
#: patch-target table); the set was empty at the controller PR, as every
#: prior series' own wiring test carried it between its task 2 and task 3.
#: That task pruned **22 of the 140**, from an ``ast``-based census (never a call-shaped
#: regex -- a bare callable passed as an argument is an ``ast.Attribute``
#: too, and the prompts series lost three names to exactly that blind spot)
#: over ``tldw_chatbook/`` + every ``Tests/`` root + ``Docs/`` + ``scripts/``
#: + ``Helper_Scripts/``, excluding only the controller module, each name's
#: own delegator body, and this file's own literal pin tuple above. The
#: other 118 KEEP: **53 unconditionally** per the recipe §4 whitelist (48
#: ``@on`` + 5 ``action_*``; media owns ZERO ``on_<message>``
#: name-dispatched handlers, so that whitelist's third member is inert
#: here) and **65 with a genuine external caller**. Prune fraction
#: 22/140 = 15.71%.
_MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset(
    {
        "_add_library_media_highlight",
        "_build_library_media_viewer_display_state",
        "_consume_library_media_find_focus",
        "_delete_library_media_highlight",
        "_dispatch_library_media_detail_request",
        "_finish_library_media_list_return",
        "_library_media_analyze_receipt_fields",
        "_library_media_console_representation",
        "_library_media_image_preview_capable",
        "_library_media_image_preview_projection",
        "_library_media_preview_request_is_current",
        "_library_media_trash_applied_scope",
        "_load_library_media_image_preview",
        "_notify_library_media_highlight_warning",
        "_notify_library_media_read_later_warning",
        "_pop_library_media_arrival_note",
        "_reload_library_media_highlights",
        "_request_library_media_sort",
        "_resolve_library_media_trash_focus_target",
        "_retry_library_media_browse",
        "_select_library_media_adjacent_item",
        "_toggle_library_media_read_later",
        # TASK-36952.3.1: 8 of its 19 movers, whose screen delegators had no
        # caller left once the move and its test retarget landed (ast census
        # over tldw_chatbook/, Tests/, Docs/, scripts/ and Helper_Scripts/).
        "_exit_library_media_viewer",
        "_library_media_find_state",
        "_library_media_reader_exit_available",
        "_open_selected_media_handoff",
        "_queue_after_library_media_viewer_recompose",
        "_reset_library_media_search_on_mode_change",
        "_set_library_media_link_receipt",
        "_toggle_library_media_find",
        # TASK-36952.3.2: 7 of its 24 movers, by the same census.
        "_begin_library_media_mutation",
        "_cancel_library_media_bulk_delete",
        "_delete_library_media_item",
        "_delete_library_media_selection",
        "_notify_library_media_edit_warning",
        "_save_library_media_edit",
        "_undo_library_media_bulk_delete",
        # TASK-36952.3.3: 7 of its 23 movers, by the same census.
        "_clear_library_media_selection_for_scope_change",
        "_library_media_type_options",
        "_notify_library_media_selection_discarded",
        "_request_library_media_type",
        "_start_library_note_from_media_source",
        "_take_library_media_note",
        "_toggle_library_media_select_mode",
        # TASK-36952.3.4: 8 of its 19 movers, by the same census.
        "_exit_library_media_trash",
        "_library_media_trash_action_disabled_reason",
        "_library_media_trash_focus_selectors",
        "_library_media_trash_retry_visible",
        "_permanently_delete_library_media_from_trash",
        "_press_library_media_trash_action",
        "_restore_library_media_from_trash",
        "_validated_library_media_trash_restore_summary",
        # TASK-36952.3.5: 9 of its 17 movers, by the same census.
        "_analyze_library_media_selection",
        "_analyze_one_library_media_item",
        "_clear_library_media_analyze_receipt",
        "_dispatch_library_media_analysis",
        "_generate_library_media_analysis",
        "_library_media_unanalyzed_ids",
        "_notify_library_media_analysis_warning",
        "_reproject_library_media_analysis_row",
        "_save_library_media_analysis",
        # TASK-36952.3.6: 8 of its 16 movers, by the same census.
        "_cache_library_media_reading_progress",
        "_fetch_library_media_reading_progress",
        "_library_media_can_rename_speakers",
        "_library_media_progress_write_is_current",
        "_library_media_speaker_rename_facts",
        "_library_media_viewer_state_cached",
        "_queue_library_media_progress_write",
        "_write_library_media_loaded_progress",
        # TASK-36952.3.7: 6 of its 23 movers, by the same census.
        "_commit_library_media_return",
        "_focus_library_media_grip_if_current",
        "_library_media_rail_return_visible",
        "_resolve_library_media_settlement_target",
        "_sync_library_media_rail_return",
        "_sync_library_media_surfaces_or_recompose",
        # TASK-36952.3.8 (the cleanup slice): 30 delegators the earlier slices
        # left with no caller through the screen. Every remaining reach is the
        # controller's own ``self.<name>``, a ``screen._media_controller`` /
        # ``LibraryMediaController`` receiver, or a fake's own attribute
        # (receiver-aware ast census over tldw_chatbook/, Tests/, scripts/ and
        # Helper_Scripts/, plus the f-string sweep). Includes the two Trash
        # statics the 3.4 review found dead. Fix round 1 re-ran the census with
        # Docs/ added and the fake rule applied consistently; it found the 8
        # at the end of this block.
        "_adopt_library_media_row_owner",
        "_advance_library_media_content_match",
        "_advance_library_media_presentation_epoch",
        "_apply_library_media_active_surface",
        "_bind_library_media_settlement_deadline",
        "_bounded_library_media_trash_title",
        "_cache_library_media_preview",
        "_cancel_library_media_trash_delete_confirmation",
        "_capture_library_media_trash_return",
        "_expire_library_media_return_settlement",
        "_fetch_library_media_highlights",
        "_focus_library_media_content_search_input",
        "_focus_library_media_page_control",
        "_focus_library_media_trash_after_paint",
        "_focus_library_media_trash_entry",
        "_focus_library_media_trash_intent",
        "_library_media_live_focus_is_allowed",
        "_library_media_reader_target_id",
        "_library_media_request_matches_current_authority",
        "_load_library_media_list_if_needed",
        "_patch_local_media_record",
        "_recompose_library_media_detail_if_unrendered",
        "_request_library_media_page",
        "_request_library_media_trash_page",
        "_schedule_library_media_image_preview",
        "_scroll_library_media_content_to_line",
        "_selected_media_handoff_payload",
        "_set_library_media_content_mode",
        "_start_library_media_read_later_toggle",
        "_valid_library_media_trash_delete_ack",
        # TASK-36952.3.8 fix round 1: 8 more by the same rule.
        "_complete_library_media_mutation",
        "_fetch_library_media_analysis_detail",
        "_focus_library_media_items_pane",
        "_library_media_mutation_summary",
        "_notify_library_media_delete_warning",
        "_request_library_media_filter",
        "_required_library_media_backing_id",
        "_sync_library_media_viewer_mutation_gate",
        # TASK-36952.3.8 fix round 2: 2 more, once a string that names the
        # delegator on a controller receiver (a ``monkeypatch.setattr`` on the
        # controller, a substring check against a controller method's source)
        # counted as a controller reference, as the header says.
        "_claim_library_media_mutation",
        "_reset_library_media_reader_to_no_selection",
    }
)

#: Phase C, task 3 (region ownership, 2026-09-09): the 16 names that left
#: ``LibraryScreen`` for a DIFFERENT reason than the 22 above, kept in their
#: own set so the two censuses stay separable. These are not zero-reference
#: dead delegators -- each is a live ``@on`` row, and 2 of them still have
#: external test callers (retargeted to the controller in the same commit).
#: They left because their message is posted INSIDE ``LibraryMediaCanvas``'s
#: own subtree, so the region widget can catch it directly and the screen's
#: routing hop is pure overhead. The behaviour did not move: each canvas row
#: forwards to the same-named controller method.
#:
#: The origin rule, and what it costs: 36 of the screen's 79 media ``@on``
#: rows are canvas-origin, but only these 16 have a controller home today.
#: The other 20 are canvas-origin with SCREEN-NATIVE bodies (outside
#: ``_MEDIA_CLUSTER_METHOD_NAMES`` entirely), and 43 are not canvas-origin at
#: all -- posted by the Reader, the Trash canvas, the Reader's content pane,
#: or the adaptive shell, which is the canvas's ancestor. The full three-way
#: partition is pinned in
#: ``Tests/UI/test_library_phase_c_region_ownership.py``.
#:
#: Effect on the counts above: the recipe §4 whitelist's unconditional keeps
#: drop from 53 to 37 (48 -> 32 ``@on``, 5 ``action_*`` unchanged -- no
#: binding moved, because Textual resolves bindings along the FOCUS chain
#: rather than by bubbling, so moving one would narrow where its key works).
#: Combined prune fraction 38/140 = 27.14%.
_MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS: frozenset[str] = frozenset(
    {
        "_handle_library_media_row_geometry_changed",
        "handle_library_media_export",
        "handle_library_media_filter_changed",
        "handle_library_media_filter_clear",
        "handle_library_media_filter_submitted",
        "handle_library_media_next",
        "handle_library_media_open_viewer",
        "handle_library_media_previous",
        "handle_library_media_retry",
        "handle_library_media_review_selected",
        "handle_library_media_review_sets",
        "handle_library_media_review_these",
        "handle_library_media_select_all",
        "handle_library_media_select_clear",
        "handle_library_media_sort",
        "handle_library_media_sort_choice",
    }
)

#: The union the delegator sweep skips: gone from the screen, for either
#: reason.
_MEDIA_CLUSTER_SCREEN_DELEGATOR_ABSENT: frozenset[str] = (
    _MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED
    | _MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS
)

#: Every name a moved body references that is NOT this controller's own
#: ``LibraryMediaState`` field and NOT another mover -- i.e. the complete
#: constructor-binding surface, derived mechanically from an ``ast`` walk of
#: all 146 moved bodies (every ``self.<attr>`` load/store, plus every
#: ``getattr(self, "<literal>")`` -- the shape recipe §3's sixth-bypass
#: entry records as invisible to a plain ``self.<attr>`` census, and the one
#: that cost the skills series a silent production regression on
#: ``focused``).
#:
#: TWO of these names appear in NO moved body at all and would be missed by
#: that walk: ``_library_canvas_projection_depth`` (read as
#: ``getattr(screen, "_library_canvas_projection_depth", 0)``) and
#: ``_library_canvas_resync_pending`` (assigned) are reached by the SHARED
#: ``_sync_library_canvas`` dispatcher through the bare ``self`` four movers
#: forward into it -- the unbound-attribute-escape shape one indirection
#: further out than the skills series' own ``focused``. Without the first,
#: ``getattr``'s default silently makes the projection-replay branch
#: unreachable forever, with no exception and no red test.
#:
#: Grouped exactly as the controller's own constructor groups them.
_MEDIA_CONTROLLER_BOUND_NAMES: tuple[str, ...] = (
    # -- framework services, live-read from the screen on every access (18)
    "app",
    "app_instance",
    "call_after_refresh",
    "call_next",
    "focused",
    "is_mounted",
    "is_running",
    "post_message",
    "query",
    "query_one",
    "refresh",
    "run_worker",
    "set_focus",
    "set_timer",
    "size",
    # TASK-36952.3.1: `_toggle_library_media_find`, a mover, calls it
    "notify",
    # TASK-36952.3.3: `action_library_media_take_note`, a mover, calls it
    "check_action",
    # TASK-36952.3.6: `_refresh_library_media_detail` and
    # `_queue_library_media_progress_write`, movers, read it
    "is_attached",
    # -- shared shell state this cluster READS (getter-only accessors) (16)
    "_library_canvas_projection_depth",
    "_library_compose_generation",
    "_library_list_entry_focus_deadline",
    "_library_list_entry_focus_generation",
    "_library_notes_compact",
    "_library_notes_focus_intent_generation",
    "_library_notes_source",
    "_library_pending_list_entry_focus",
    "_library_pending_list_entry_focus_anchor",
    "_library_pending_list_entry_media_return",
    "_local_source_records",
    # TASK-36952.3.2: the three moved delete/Undo bodies adjust the Media count
    "_local_source_counts",
    # TASK-36952.3.3: `_take_library_media_note` reads the Notes controller and
    # writes a field on the Notes state object
    "_notes_controller",
    "_notes_state",
    # TASK-36952.3.6: `_refresh_library_media_detail` fences on it
    "_library_snapshot_state_generation",
    # TASK-36952.3.8 (ruling R4): `_sync_library_media_viewer_state` gates the
    # TASK-38981 stage-pick landing on it, which it read through
    # `self._screen` until now
    "_library_stage_pick_landing",
    # task-32350 (critique #10 review, finding 10) read `_library_loaded`,
    # `_library_lookup_error`, `_local_source_total_known` and
    # `_local_source_counts` through `self._screen` in
    # `_library_media_unfiltered_total`. TASK-36952.3.8 replaced those four
    # reaches with the named `_library_source_total` shell helper below, and
    # `test_media_controller_takes_a_typed_host_instead_of_the_screen` now
    # fails on any screen read that is not a host framework service.
    # -- shared shell state this cluster also WRITES (getter + setter) (7)
    "_library_canvas_resync_pending",
    "_library_list_entry_focus_timer",
    "_library_notes_programmatic_focus_target",
    "_library_notes_restoring_focus",
    "_library_selected_row_id",
    # TASK-36952.3.6: `_refresh_library_media_detail` sets it
    "_library_entry_reconcile_dirty",
    # TASK-36952.3.7: the review-set Undo and receipt-close handlers read and
    # set it; the screen's Undo worker clears it
    "_review_dismiss_undo_in_flight",
    # -- the 2 prior-extracted media WIRING controller instances (2)
    "_library_media_browse_controller",
    "_library_media_trash_browse_controller",
    # -- general Library-wide shell helpers, named constructor callables (43)
    "_acknowledge_library_destination_change",
    "_active_review_set_banner",
    "_apply_library_open_item_surface",
    "_arm_library_list_entry_focus",
    "_disarm_library_list_entry_focus",
    "_focus_library_control",
    "_focus_library_list_entry",
    "_library_entry_route_key",
    "_open_library_export_canvas",
    "_open_library_item_by_id",
    "_register_footer_shortcuts",
    "_review_dismiss_receipt_name",
    "_review_selected_worker",
    "_review_set_picker_worker",
    "_review_these_worker",
    "_run_library_service_call",
    "_safe_text",
    "_source_record_id",
    "_sync_library_conversation_reader_layout_from_shell",
    "_sync_library_notes_reader_layout_from_shell",
    "_sync_library_prompts_reader_layout_from_shell",
    "_sync_library_skills_reader_layout_from_shell",
    "_walk_active_review_set",
    "request_library_reader_layout_refresh",
    # TASK-36952.3.1: shell helpers its movers call (24 -> 29)
    "_close_open_library_choice_strip",
    "_focus_library_rail_action",
    "_invalidate_library_workspace_depth_state",
    "_library_open_choice_strip",
    "_library_workspace_depth_state",
    # TASK-36952.3.3: shell helpers its movers call (29 -> 33)
    "_arm_stage_pick_landing",
    "_cancel_pending_review_set_resume",
    "_focus_library_list_entry_if_current",
    "_select_library_rail_row",
    # TASK-36952.3.5: the shell helper its movers call (33 -> 34)
    "_update_library_ingest_dynamic_regions",
    # TASK-36952.3.6: shell helpers its movers call (34 -> 36)
    "_review_set_service",
    "_schedule_library_entry_reconcile",
    # TASK-36952.3.7: shell helpers its movers call (36 -> 41)
    "_library_adaptive_reader_allocation_is_current",
    "_notify_review_set",
    "_review_dismiss_undo_worker",
    "_toggle_reviewed_unguarded",
    "action_library_narrow_stage_return",
    # TASK-36952.3.8 (ruling R4): the two shell helpers that replace the
    # controller's direct screen reaches (41 -> 43)
    "_land_stage_pick",
    "_library_source_total",
    # -- named late-binding callables for the excluded media methods a mover
    #    still calls: 35 at wave 7; TASK-36952.3.1 moved 5 of them here: 30;
    #    TASK-36952.3.2 moved 2 more: 28; TASK-36952.3.3 moved 6 more: 22;
    #    TASK-36952.3.4 moved 5 more: 17; TASK-36952.3.5 moved 2 more: 15;
    #    TASK-36952.3.6 moved 5 more: 10; TASK-36952.3.7 moved 9 more: 1;
    #    TASK-36952.3.8 moved the last, ``_sanitize_media_field``, here as a
    #    staticmethod (ruling R3): 0.
)


@pytest.mark.unit
def test_media_cluster_method_names_are_genuinely_media_named() -> None:
    """Guards the hand-kept cluster list against drift with the census:
    every name must contain "media" (case-insensitive) -- a typo here would
    silently test the wrong surface.
    """
    not_media_named = [
        n for n in _MEDIA_CLUSTER_METHOD_NAMES if "media" not in n.lower()
    ]
    assert not not_media_named, f"non-media-named cluster entries: {not_media_named!r}"
    # 2026-10-20, TASK-34000.18: 140 -> 141, ``_library_media_reader_target_id``
    # (the one identity every Reader-scoped action reads), born on the
    # controller with its one-line screen delegator. Fix round 1: 141 -> 142,
    # ``_adopt_library_media_highlight`` (the one writer of the list's
    # resolved highlight into the selection) moved onto the controller so
    # ``_build_library_media_active_child`` can route through it too.
    # 2026-10-20, TASK-34000.22: 142 -> 146, the two authoring-form draft
    # recorders (``handle_library_media_edit_field_changed``,
    # ``handle_library_media_analysis_text_changed``), the one dirty
    # predicate every exit reads (``_library_media_editor_dirty``) and its
    # veto toast (``_notify_media_editor_dirty_veto``), each with a one-line
    # screen delegator. Fix round 2 (2026-10-21): 146 -> 147,
    # ``_reveal_library_media_dirty_form`` (the refusal's and the Media
    # press's "put the form on screen": viewer view + the form's tab).
    # 2026-11-15, TASK-36952.3.1: 147 -> 166, the 19 find / Escape label /
    # viewer-back / Console handoff bodies moved off the screen.
    # 2026-11-15, TASK-36952.3.2: 166 -> 190, the 24 mutation-interlock,
    # delete, Undo, export and edit-save bodies.
    # 2026-11-15, TASK-36952.3.3: 190 -> 213, the 23 select-mode, browse,
    # type-filter and take-note bodies.
    # 2026-11-15, TASK-36952.3.4: 213 -> 232, the 19 Trash bodies.
    # 2026-11-15, TASK-36952.3.5: 232 -> 249, the 17 analysis and bulk
    # Analyze bodies.
    # 2026-11-15, TASK-36952.3.6: 249 -> 265, the 16 detail, reading-progress,
    # speaker-rename and review-decoration bodies.
    # 2026-11-17, TASK-36952.3.7: 265 -> 288, the 23 return-settlement,
    # reader-layout, rail-return and review-action bodies.
    # 2026-11-17, TASK-36952.3.8: 288 -> 289, ``_sanitize_media_field``
    # (ruling R3).
    assert len(_MEDIA_CLUSTER_METHOD_NAMES) == 289, (
        f"expected 289 moved names, got {len(_MEDIA_CLUSTER_METHOD_NAMES)}"
    )
    assert len(set(_MEDIA_CLUSTER_METHOD_NAMES)) == len(_MEDIA_CLUSTER_METHOD_NAMES), (
        "duplicate entries in _MEDIA_CLUSTER_METHOD_NAMES"
    )


@pytest.mark.unit
def test_media_controller_owns_its_cluster() -> None:
    """Every one of the 142 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    ``test_prompts_controller_owns_its_cluster``.
    """
    from tldw_chatbook.UI.Library_Modules.library_media_controller import (
        LibraryMediaController,
    )

    missing = [
        name
        for name in _MEDIA_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryMediaController, name, None))
    ]
    assert not missing, f"LibraryMediaController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_media_handlers() -> None:
    """Every one of the 142 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 3
    staticmethods, to the module-level controller CLASS) -- unless a later
    cleanup task pruned it.

    A same-name forwarding check, not a loose "the controller is referenced
    somewhere" substring check. Skips the names in
    ``_MEDIA_CLUSTER_SCREEN_DELEGATOR_ABSENT`` -- task 3's zero-reference
    census plus phase C's 16 canvas-owned rows -- and instead asserts each
    such name is genuinely ABSENT from ``LibraryScreen``, so a future
    accidental re-add fails loudly here rather than silently reintroducing
    dead code (or, for the phase-C 16, a second receiver for a message the
    canvas already handles).
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _MEDIA_CLUSTER_METHOD_NAMES:
        if name in _MEDIA_CLUSTER_SCREEN_DELEGATOR_ABSENT:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen but is back -- either "
                "wire it as a delegator again or drop it from "
                "_MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED / "
                "_MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(rf"_media_controller\.{escaped}\(", src) and not re.search(
            rf"LibraryMediaController\.{escaped}\(", src
        ):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_the_canvas_owned_rows_are_movers_that_landed_on_the_canvas() -> None:
    """The phase-C set must stay a strict, disjoint subset of the movers.

    Two ways this could rot silently: a name could be added that was never a
    mover (so the delegator sweep would skip a row it never checked), or a
    name could sit in BOTH absence sets with two different stated reasons.
    Neither is caught by the sweep itself, because both make it skip more.
    """
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas

    assert len(_MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS) == 16
    assert _MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS <= set(
        _MEDIA_CLUSTER_METHOD_NAMES
    ), "a canvas-owned row is not one of the 142 moved names"
    assert not (
        _MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS
        & _MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED
    ), "a name claims both absence reasons"
    missing = sorted(
        name
        for name in _MEDIA_CLUSTER_SCREEN_ROWS_OWNED_BY_THE_CANVAS
        if getattr(LibraryMediaCanvas, name, None) is None
    )
    assert not missing, (
        f"left the screen but never arrived on LibraryMediaCanvas: {missing!r}"
    )


@pytest.mark.unit
def test_media_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """Each cluster staticmethod the screen still offers forwards to the CLASS.

    Five cluster names are staticmethods: two keep a screen delegator
    (``_restore_library_media_scope``, ``_sanitize_media_field``) and three
    were pruned (``_bounded_library_media_trash_title``,
    ``_valid_library_media_trash_delete_ack``,
    ``_validated_library_media_trash_restore_summary``), which must stay
    absent. A ``@staticmethod`` has no ``self`` to reach ``self._media_controller``
    through, so its delegator names the module-level controller class
    directly -- the conversations exemplar's own corrected shape (recipe
    §11, "the static-method delegator pattern"), reused unchanged by every
    series since.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert _MEDIA_CLUSTER_STATICMETHOD_NAMES <= set(_MEDIA_CLUSTER_METHOD_NAMES), (
        "a staticmethod name is not in the mover tuple"
    )
    not_class_forwarding = []
    for name in _MEDIA_CLUSTER_STATICMETHOD_NAMES:
        if name in _MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen but is back"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_class_forwarding.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        if not re.search(rf"LibraryMediaController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_media_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    The moved bodies spell the ORIGINAL flat names, so each one has to keep
    resolving on the controller -- through the same single-source
    ``media_state_shim_attr()`` the screen's own task-1 shim block uses.
    """
    from tldw_chatbook.UI.Library_Modules.library_media_controller import (
        LibraryMediaController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryMediaState)}
    assert field_names, "state object is empty"
    missing = []
    for name in sorted(field_names):
        shim_attr = media_state_shim_attr(name)
        if not isinstance(getattr(LibraryMediaController, shim_attr, None), property):
            missing.append(shim_attr)
    assert not missing, (
        f"no media controller shim property found for state field(s): {missing!r}"
    )


@pytest.mark.unit
def test_media_controller_binds_every_name_its_moved_bodies_use() -> None:
    """Constructor-binding coverage: the byte-for-byte canon's own contract.

    A moved body is never edited, so every non-state name it spells as
    ``self.<name>`` (or reaches by ``getattr(self, "<literal>")``, or hands
    to the shared ``_sync_library_canvas`` dispatcher as bare ``self``) has
    to keep resolving -- on the CONTROLLER now, not the screen. This asserts
    the resolution exists at the CLASS level rather than on a constructed
    instance, deliberately: ``workers``-shaped framework forwards raise off
    the app tree, so an instance probe would report a false failure for a
    binding that is in fact present and correct.
    """
    from tldw_chatbook.UI.Library_Modules.library_media_controller import (
        LibraryMediaController,
    )

    # 2026-11-15, TASK-36952.3.1: 92 -> 93. Five late-binding callbacks became
    # moved methods (-5); `notify` and five shell helpers were bound (+6).
    # 2026-11-15, TASK-36952.3.2: 93 -> 92. Two late-binding callbacks became
    # moved methods (-2); the `_local_source_counts` getter was bound (+1).
    # 2026-11-15, TASK-36952.3.3: 92 -> 93. Six pinned late-binding callbacks
    # became moved methods (-6); `check_action`, the `_notes_controller` and
    # `_notes_state` getters and four shell helpers were bound (+7).
    # 2026-11-15, TASK-36952.3.4: 93 -> 88. Five pinned late-binding callbacks
    # became moved methods (-5); nothing new was bound.
    # 2026-11-15, TASK-36952.3.5: 88 -> 87. Two pinned late-binding callbacks
    # became moved methods (-2); one shell helper was bound (+1).
    # 2026-11-15, TASK-36952.3.6: 87 -> 87. Five pinned late-binding callbacks
    # became moved methods (-5); `is_attached`, the
    # `_library_snapshot_state_generation` getter, the
    # `_library_entry_reconcile_dirty` getter and setter and two shell helpers
    # were bound (+5).
    # 2026-11-17, TASK-36952.3.7: 87 -> 84. Nine pinned late-binding callbacks
    # became moved methods (-9); five shell helpers and the
    # `_review_dismiss_undo_in_flight` getter and setter were bound (+6).
    # 2026-11-17, TASK-36952.3.8: 84 -> 83. The last late-binding callback,
    # `_sanitize_media_field`, became a staticmethod on the controller (-1).
    # Then 83 -> 86: the controller's direct screen reaches became named
    # dependencies, the `_library_stage_pick_landing` getter and the
    # `_land_stage_pick` and `_library_source_total` shell helpers (+3).
    assert len(_MEDIA_CONTROLLER_BOUND_NAMES) == 86, (
        f"expected 86 bound names, got {len(_MEDIA_CONTROLLER_BOUND_NAMES)}"
    )
    assert len(set(_MEDIA_CONTROLLER_BOUND_NAMES)) == len(
        _MEDIA_CONTROLLER_BOUND_NAMES
    ), "duplicate entries in _MEDIA_CONTROLLER_BOUND_NAMES"
    unbound = [
        name
        for name in _MEDIA_CONTROLLER_BOUND_NAMES
        if not isinstance(getattr(LibraryMediaController, name, None), property)
    ]
    assert not unbound, (
        "moved bodies reference these names, but the controller binds no "
        f"property for them: {unbound!r}"
    )


def _media_host_members() -> set[str]:
    tree = ast.parse(_MEDIA_CONTROLLER_PATH.read_text(encoding="utf-8"))
    host = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LibraryMediaHost"
    )
    return {node.name for node in host.body if isinstance(node, ast.FunctionDef)}


@pytest.mark.unit
def test_media_controller_takes_a_typed_host_instead_of_the_screen() -> None:
    """TASK-36952.3.8, coordinator ruling R4 (parent AC #2).

    The controller's DOM handle is the ``LibraryMediaHost`` Protocol, its
    module never imports ``LibraryScreen``, and everything it reads through
    that handle is one of the Protocol's framework services: the shell state
    and shell helpers it needs are named constructor dependencies. The handle
    is also the identity the four ``<widget>.ancestors`` checks test, so a
    bare ``self._screen`` may appear only in an ``in``/``not in`` test.
    """
    tree = ast.parse(_MEDIA_CONTROLLER_PATH.read_text(encoding="utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    members = _media_host_members()

    imports = [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (
            (node.module or "").endswith("library_screen")
            or any(alias.name == "LibraryScreen" for alias in node.names)
        )
    ]
    assert not imports, f"the media controller imports LibraryScreen: {imports}"
    names = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "LibraryScreen"
    ]
    assert not names, f"the media controller names LibraryScreen at lines {names}"

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LibraryMediaController"
    )
    init = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    handle = init.args.args[1]
    assert handle.annotation is not None
    assert ast.unparse(handle.annotation).strip("\"'") == "LibraryMediaHost"

    reaches = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and node.attr == "_screen"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            continue
        parent = parents[node]
        if isinstance(parent, ast.Attribute):
            if parent.attr not in members:
                reaches.append(f"line {parent.lineno}: self._screen.{parent.attr}")
        elif isinstance(parent, ast.Compare) and all(
            isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops
        ):
            continue
        elif isinstance(parent, ast.Assign) and node in parent.targets:
            continue
        else:
            reaches.append(f"line {node.lineno}: {ast.unparse(parent)[:80]}")
    assert not reaches, (
        "the media controller reads the screen beyond its host's framework "
        f"services: {reaches}"
    )


#: TASK-36952.3.8 (ruling R4): the host's members, pinned. Adding a member
#: is the one way to widen what the controller may read off the screen, so
#: it must change this set and pass the framework check below.
_MEDIA_HOST_MEMBERS: frozenset[str] = frozenset(
    {
        "app",
        "app_instance",
        "call_after_refresh",
        "call_next",
        "check_action",
        "focus_chain",
        "focused",
        "is_attached",
        "is_mounted",
        "is_running",
        "notify",
        "post_message",
        "query",
        "query_one",
        "refresh",
        "run_worker",
        "set_focus",
        "set_timer",
        "size",
    }
)


@pytest.mark.unit
def test_library_screen_implements_the_media_host() -> None:
    """The host offers framework services only, and ``LibraryScreen`` has them.

    The member set is pinned (19). Every member but ``app_instance`` must
    resolve on Textual's own ``Screen``, so a ``LibraryScreen`` attribute
    (a shell field or helper) cannot be added to the Protocol and read
    through the host. ``app_instance`` is the one exception: an instance
    attribute that ``BaseAppScreen.__init__`` sets and ``LibraryScreen``
    inherits.
    """
    from textual.screen import Screen

    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    members = _media_host_members()
    assert members == _MEDIA_HOST_MEMBERS, (
        f"host members drifted: added {sorted(members - _MEDIA_HOST_MEMBERS)}, "
        f"removed {sorted(_MEDIA_HOST_MEMBERS - members)}"
    )
    not_framework = sorted(
        name for name in members - {"app_instance"} if not hasattr(Screen, name)
    )
    assert not not_framework, (
        f"host member(s) that are not Textual Screen services: {not_framework}"
    )
    assert issubclass(LibraryScreen, BaseAppScreen)
    init = ast.parse(inspect.getsource(BaseAppScreen.__init__).lstrip()).body[0]
    assigns_app_instance = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "app_instance"
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            for target in node.targets
        )
        for node in ast.walk(init)
    )
    assert assigns_app_instance, "BaseAppScreen.__init__ no longer sets app_instance"


@pytest.mark.unit
def test_no_media_controller_argument_calls_a_media_named_screen_method() -> None:
    """TASK-36952.3.8 AC #3: under ruling R4 the screen passes no callback into
    a media-named ``LibraryScreen`` method. Getters of shell state
    (``self._media_state``, the two media wiring controllers) are not
    methods; the shell, review-set, notes and workspace callbacks are not
    media behaviour.
    """
    tree = ast.parse(_LIBRARY_SCREEN_PATH.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LibraryScreen"
    )
    methods = {
        node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    call = next(
        node
        for node in ast.walk(cls)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LibraryMediaController"
    )
    media_callbacks = sorted(
        f"{keyword.arg}: self.{node.attr}"
        for keyword in call.keywords
        for node in ast.walk(keyword.value)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in methods
        and "media" in node.attr.lower()
    )
    assert not media_callbacks, (
        f"media controller arguments still call media-named screen methods: {media_callbacks}"
    )
