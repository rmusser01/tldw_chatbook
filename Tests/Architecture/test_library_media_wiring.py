"""Media extraction series: state object exists and is screen-wired.

Wave-7 Task 1 (media series 1/3, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; prompts series precedent: ``Tests/
Architecture/test_library_prompts_wiring.py``, its state-PR-era shape --
the closest match, since Media likewise carries a bare-underscore
third-prefix field and an entangled reader-preferences group). Every field
``LibraryMediaState`` declares must have a matching generated property shim
on ``LibraryScreen``, resolved via ``media_state_shim_attr()`` -- the
single-source two-way prefix mapping (``_library_media_`` default, bare
``_`` for the one unprefixed field, ``selected_media_id``) documented in
``library_media_state.py``'s own module docstring. Unlike a looser "either
prefix works" check, these tests assert the EXACT expected shim name per
field, and that each shim genuinely reads AND writes the state object rather
than merely existing as a property.

**What the per-field sweep can and cannot prove.** Because the screen's shim
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
checks (``_MEDIA_CLUSTER_METHOD_NAMES``, 146 names) plus a
constructor-binding coverage check (``_MEDIA_CONTROLLER_BOUND_NAMES``, 88
names) carried from the prompts series, which added it because the skills
series shipped a silent production regression precisely in that gap (a moved
body's ``getattr(self, "focused", None)`` with no ``focused`` property
bound; recipe §3's unbound-attribute-escape entry). Media's own instance of
that shape is one indirection further out and is called out on the tuple
itself. See ``library_media_controller.py``'s own module docstring for the
full 251-candidate derivation, the 105 exclusions, and the
connected-components evidence behind the single-controller decision.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_media_state import (
    MEDIA_UNPREFIXED_STATE_FIELDS,
    LibraryMediaState,
    media_state_shim_attr,
)

#: The ownership census's own MOVE count (recipe §2 script, substring
#: "media" over every ``__init__``-stored attribute of ``LibraryScreen``,
#: plus a full class-body ``Assign``/``AnnAssign`` scan that found the one
#: class-level-only attribute, ``_library_media_arrival_note``): 85
#: attributes found, 2 WIRING (live controller instances -- see below), 1
#: BLOCKED (``_library_pending_list_entry_media_return``, shared shell
#: state -- see the state module's docstring), so 82 move. Pinned here so a
#: field silently added to or dropped from the dataclass fails loudly
#: instead of quietly resizing the shim surface this file checks.
_EXPECTED_MEDIA_STATE_FIELD_COUNT = 82

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
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryMediaState)}
    assert field_names, "state object is empty"
    assert len(field_names) == _EXPECTED_MEDIA_STATE_FIELD_COUNT, (
        f"expected {_EXPECTED_MEDIA_STATE_FIELD_COUNT} media fields, "
        f"got {len(field_names)}"
    )
    missing = []
    for name in sorted(field_names):
        shim_attr = media_state_shim_attr(name)
        if not isinstance(getattr(LibraryScreen, shim_attr, None), property):
            missing.append(shim_attr)
    assert not missing, f"no screen shim property found for: {missing!r}"


@pytest.mark.unit
def test_every_shim_reads_and_writes_its_own_state_field() -> None:
    """Each generated property is a real two-way shim, not a stub.

    A getter/setter pair that existed but bound the WRONG field (the
    closure-binding trap a `for` loop over `dataclasses.fields` invites --
    every generated property capturing the LAST field unless the name is
    bound as a default argument) would satisfy a bare `isinstance(...,
    property)` check while silently aliasing 82 names onto one field. This
    round-trips a distinct sentinel through every name to rule that out.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    screen = object.__new__(LibraryScreen)
    state = LibraryMediaState()
    screen._media_state = state

    field_names = sorted(f.name for f in dataclasses.fields(LibraryMediaState))
    read_mismatch = []
    write_mismatch = []
    for name in field_names:
        shim_attr = media_state_shim_attr(name)
        if getattr(screen, shim_attr) is not getattr(state, name):
            read_mismatch.append(shim_attr)
        sentinel = object()
        setattr(screen, shim_attr, sentinel)
        if getattr(state, name) is not sentinel:
            write_mismatch.append(shim_attr)
        if getattr(screen, shim_attr) is not sentinel:
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
#: (no property/setter-pair gap) -- minus 105 exclusions: 73 unbound-fake-
#: self, 16 instance-attribute-monkeypatch, 8 source-census, 4
#: module-globals-coupling, 2 class-monkeypatch, 1 shared-shell-helper
#: (``_sanitize_media_field``, which the Prompts and Notes clusters also
#: call) and 1 generic dispatcher (``_toggle_library_media_reader_pane``).
#: NOT a prefix/substring shortcut. See ``library_media_controller.py``'s
#: module docstring for the full per-name reasoning behind every exclusion,
#: and for the connected-components evidence behind the single-controller
#: decision.
_MEDIA_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_add_library_media_highlight",
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
    "_commit_library_media_return",
    "_consume_library_media_find_focus",
    "_delete_library_media_highlight",
    "_disarm_library_media_return_for_route_change",
    "_dispatch_library_media_detail_request",
    "_exit_library_media_viewer",
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
    "_library_media_focus_target_matches_receipt",
    "_library_media_image_preview_capable",
    "_library_media_image_preview_projection",
    "_library_media_item_traversal_active",
    "_library_media_list_surface_active",
    "_library_media_live_focus_is_allowed",
    "_library_media_preview_request_is_current",
    "_library_media_reader_identity",
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
    "_resolve_library_media_settlement_target",
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
    "_stop_library_media_filter_timer",
    "_stop_library_media_selection_debounce",
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
    }
)

#: Filled in by this series' own cleanup task (task 3, media series 3/3):
#: the moved names whose screen delegator has ZERO references outside its
#: own body anywhere in the repo, across all four census spellings
#: (attribute, quoted-string, bare-assignment, patch-target table). Empty
#: at the controller PR, exactly as every prior series' own wiring test
#: carried it between its task 2 and task 3.
_MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset()

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
    # -- framework services, live-read from the screen on every access (15)
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
    # -- shared shell state this cluster READS (getter-only accessors) (11)
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
    # -- shared shell state this cluster also WRITES (getter+setter) (5)
    "_library_canvas_resync_pending",
    "_library_list_entry_focus_timer",
    "_library_notes_programmatic_focus_target",
    "_library_notes_restoring_focus",
    "_library_selected_row_id",
    # -- the 2 prior-extracted media WIRING controller instances (2)
    "_library_media_browse_controller",
    "_library_media_trash_browse_controller",
    # -- general Library-wide shell helpers, named constructor callables (24)
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
    # -- named late-binding callables for the excluded media methods a mover still calls (31)
    "_arm_library_media_return_settlement",
    "_build_library_media_trash_state",
    "_cancel_library_media_bulk_delete",
    "_capture_library_media_loaded_progress",
    "_clear_library_media_selection_for_scope_change",
    "_close_library_media_find",
    "_exit_library_media_trash",
    "_library_media_analysis_provider_reason",
    "_library_media_backing_id",
    "_library_media_content_matches",
    "_library_media_content_signature",
    "_library_media_layout_signature",
    "_library_media_reader_exit_available",
    "_library_media_return_candidate",
    "_library_media_settlement_tree",
    "_library_media_trash_focus_selectors",
    "_library_media_trash_retry_visible",
    "_library_media_type_options",
    "_library_media_viewer_state_cached",
    "_open_selected_media_handoff",
    "_reconcile_library_media_stage_presentation",
    "_refresh_library_media_detail",
    "_request_library_media_browse",
    "_reset_library_media_search_on_mode_change",
    "_restore_library_media_loaded_progress",
    "_sanitize_media_field",
    "_save_library_media_analysis",
    "_sync_library_media_browse_state",
    "_sync_library_media_reader_layout_from_shell",
    "_sync_library_media_trash_state",
    "_toggle_library_media_select_mode",
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
    assert len(_MEDIA_CLUSTER_METHOD_NAMES) == 146, (
        f"expected 146 moved names, got {len(_MEDIA_CLUSTER_METHOD_NAMES)}"
    )
    assert len(set(_MEDIA_CLUSTER_METHOD_NAMES)) == len(_MEDIA_CLUSTER_METHOD_NAMES), (
        "duplicate entries in _MEDIA_CLUSTER_METHOD_NAMES"
    )


@pytest.mark.unit
def test_media_controller_owns_its_cluster() -> None:
    """Every one of the 146 moved names is a callable on the controller.

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
    """Every one of the 146 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 3
    staticmethods, to the module-level controller CLASS) -- unless a later
    cleanup task pruned it.

    A same-name forwarding check, not a loose "the controller is referenced
    somewhere" substring check. Skips the names in
    ``_MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED`` (task 3's census) and instead
    asserts each such name is genuinely ABSENT from ``LibraryScreen``, so a
    future accidental re-add fails loudly here rather than silently
    reintroducing dead code.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _MEDIA_CLUSTER_METHOD_NAMES:
        if name in _MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen but is back -- either "
                "wire it as a delegator again or drop it from "
                "_MEDIA_CLUSTER_SCREEN_DELEGATOR_PRUNED"
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
def test_media_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 3 staticmethod names in the cluster forward to the CLASS.

    A ``@staticmethod`` has no ``self`` to reach ``self._media_controller``
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

    assert len(_MEDIA_CONTROLLER_BOUND_NAMES) == 88, (
        f"expected 88 bound names, got {len(_MEDIA_CONTROLLER_BOUND_NAMES)}"
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
