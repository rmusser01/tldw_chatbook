"""Notes extraction series: state object exists and is screen-wired.

Wave-8 Task 1 (notes series 1/3, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; media series precedent: ``Tests/
Architecture/test_library_media_wiring.py``, its state-PR-era shape -- the
closest match, since Notes likewise carries a bare-underscore extra-prefix
field and an entangled reader-preferences group). Every field
``LibraryNotesState`` declares must have a matching generated property shim on
``LibraryScreen``, resolved via ``notes_state_shim_attr()`` -- the
single-source four-way prefix mapping (``_library_notes_`` default,
``_library_note_`` singular, ``_library_file_notes_`` for the Folder-Files
group, bare ``_`` for the one unprefixed field, ``selected_note_id``)
documented in ``library_notes_state.py``'s own module docstring. Unlike a
looser "any prefix works" check, these tests assert the EXACT expected shim
name per field, and that each shim genuinely reads AND writes the state object
rather than merely existing as a property.

**What the per-field sweep can and cannot prove.** Because the screen's shim
loop and this file's sweep both call ``notes_state_shim_attr()``, the sweep
proves the two are CONSISTENT -- it cannot prove the mapping is CORRECT. The
prompts series proved that hole is real: deleting a prefix branch from the
shared resolver outright left its sweep passing, because generator and test
then agreed on the same wrong answer. ``test_notes_state_shim_attr_maps_
each_prefix_family_to_its_literal_name`` below closes it with hard-coded
expected strings -- one per prefix family, including the bare-underscore one
-- and is carried here from BIRTH rather than added by a later review round.
Notes has FOUR families, one more than any prior subsystem, which is exactly
the condition that makes the hole cheap to fall into.

The state module under test is ``tldw_chatbook.UI.Library_Modules.library_
notes_state`` -- NOT the unrelated, pre-existing ``tldw_chatbook.Library.
library_notes_state`` domain module of the same basename (see that docstring's
own note on the collision).

**Task 2 (controller PR)** adds the full-cluster ownership / same-name-
delegator-forwarding / staticmethod-class-forwarding / controller-state-shim
checks (``_NOTES_CLUSTER_METHOD_NAMES``, 185 names) plus a
constructor-binding coverage check (``_NOTES_CONTROLLER_BOUND_NAMES``, 102
names) carried from the prompts and media series, which added it because the
skills series shipped a silent production regression precisely in that gap (a
moved body's ``getattr(self, "focused", None)`` with no ``focused`` property
bound; recipe SS3's unbound-attribute-escape entry). It also adds the
``on_<message>`` whitelist resolution recipe SS4's third member requires --
run against Textual's own NAME-based dispatch rather than against a reference
census. See ``library_notes_controller.py``'s own module docstring for the
full 285-candidate derivation, the 100 exclusions, and the
connected-components evidence behind the single-controller decision.

**Task 3 (cleanup PR)** deleted the screen's generated shim block, so the
per-field sweep above now runs the OTHER way: ``test_the_screen_no_longer_
carries_a_notes_state_shim`` asserts ABSENCE on ``LibraryScreen``, and the
read/write round-trip that proved the generated properties are real two-way
shims lives on in ``test_every_controller_shim_reads_and_writes_its_own_
state_field``, aimed at ``LibraryNotesController``'s own permanent copy of the
identical loop -- which carries the identical closure-binding trap and is what
keeps the 185 moved bodies byte-for-byte. It also filled
``_NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED`` with the 26 names whose screen
delegator had zero references across all SIX census spellings. TASK-31932
step 68 retargets one test-only caller to the existing owner (27 total).
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_notes_state import (
    FILE_NOTES_STATE_FIELDS,
    NOTE_SINGULAR_STATE_FIELDS,
    NOTE_UNPREFIXED_STATE_FIELDS,
    LibraryNotesState,
    notes_state_shim_attr,
)

#: The ownership census's own MOVE count (recipe §2 script, substring "note"
#: over every ``__init__``-stored attribute of ``LibraryScreen``, plus a full
#: class-body ``Assign``/``AnnAssign`` scan, which found no class-level-only
#: notes attribute): **105** attributes found, 3 WIRING (live prior-extracted
#: coordinator instances -- see below), 2 BLOCKED (shared shell focus state a
#: SECOND subsystem writes -- see below), so 100 move. Pinned here so a field
#: silently added to or dropped from the dataclass fails loudly instead of
#: quietly resizing the shim surface this file checks.
#:
#: 100 at the wave-8 census; **104** on the merged riders wave: task-32144
#: added the Trash view's ``trash`` snapshot (1) and task-32145 added
#: ``backlinks`` plus ``backlinks_status`` (2). All three are ordinary
#: Notes-owned ``_library_notes_``-prefixed state, so they extend the census
#: rather than change its shape. task-32100 adds a fourth,
#: ``navigation_focus_intent`` -- whether the locator behind
#: ``navigation_status`` will take focus when it lands. (task-32144
#: briefly also had
#: ``trash_loading``; the exclusive worker group already supersedes an
#: in-flight read, and a flag that DROPPED the newer request was a bug, not a
#: guard -- PR #2553 review.) **105** on the wave-3 editor-keys branch:
#: task-32268 adds ``delete_origin_scroll``, Info's scroll offset when the
#: delete prompt opens, restored on cancel (104 + 1 = 105).
# PR-2427 removes the inert auto_sync_timer in its new state owner (TASK-31909).
_EXPECTED_NOTES_STATE_FIELD_COUNT = 104

#: The 3 WIRING attributes the state PR deliberately left on ``LibraryScreen``
#: (the ``_conversation_reader_controller``/``_library_media_browse_
#: controller`` precedent): each holds a live prior-extracted coordinator
#: instance constructed with callables that close over the screen, not data.
_NOTES_WIRING_SCREEN_ATTRS: tuple[str, ...] = (
    "_library_note_import_controller",
    "_library_notes_sync_controller",
    "_library_note_session",
)

#: The 2 BLOCKED attributes. Both carry a notes name and Notes' own methods do
#: write them -- but so does MEDIA, from bodies that have already left this
#: screen: ``library_media_controller.py`` binds each with a getter AND a
#: setter accessor and names them among the "5 shared shell state [names] this
#: cluster also WRITES", beside ``_library_selected_row_id``. A field a second
#: subsystem WRITES is shared shell state by the recipe's >=2-subsystems rule.
_NOTES_BLOCKED_SCREEN_ATTRS: tuple[str, ...] = (
    "_library_notes_programmatic_focus_target",
    "_library_notes_restoring_focus",
)


@pytest.mark.unit
def test_state_object_declares_the_censused_field_count() -> None:
    """The half of the old shim-surface pin that never needed the screen.

    (wave-8 task 3.) `test_state_object_fields_match_the_shim_surface` asserted
    both that the dataclass declares the censused 100 fields AND that every one
    of them has a generated `@property` on `LibraryScreen`. The second half is
    now the OPPOSITE invariant -- see
    `test_the_screen_no_longer_carries_a_notes_state_shim` -- and the
    controller's own permanent loop is pinned by
    `test_notes_controller_exposes_every_state_field` and
    `test_every_controller_shim_reads_and_writes_its_own_state_field`.
    """
    field_names = {f.name for f in dataclasses.fields(LibraryNotesState)}
    assert field_names, "state object is empty"
    assert len(field_names) == _EXPECTED_NOTES_STATE_FIELD_COUNT, (
        f"expected {_EXPECTED_NOTES_STATE_FIELD_COUNT} notes fields, "
        f"got {len(field_names)}"
    )


@pytest.mark.unit
def test_the_screen_no_longer_carries_a_notes_state_shim() -> None:
    """The cleanup PR's own inverse pin: ABSENCE, asserted by name.

    (wave-8 task 3.) The generated block between the BEGIN/END sentinels in
    `library_screen.py` is deleted; every screen-side reference reads
    `self._notes_state.<field>` directly. A re-introduced shim (or a stray
    same-named property arriving from anywhere else) fails here rather than
    silently restoring the flat surface this series exists to remove.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    resurrected = [
        notes_state_shim_attr(f.name)
        for f in dataclasses.fields(LibraryNotesState)
        if isinstance(
            getattr(LibraryScreen, notes_state_shim_attr(f.name), None), property
        )
    ]
    assert not resurrected, (
        "LibraryScreen carries flat notes-state shim properties again: "
        f"{resurrected!r}"
    )


@pytest.mark.unit
def test_notes_state_shim_attr_maps_each_prefix_family_to_its_literal_name() -> None:
    """The four-way mapping, pinned against LITERAL expected strings.

    The only assertion in this file that is not self-referential: every other
    check resolves the expected name by calling the same
    `notes_state_shim_attr()` the screen's own shim loop calls, so screen and
    test agree even when the mapping is wrong. The prompts series proved that
    hole empirically (deleting its plural branch left the rest of its wiring
    file green), which is why this one ships from birth. One field per prefix
    family, spelled out:
    """
    # default family -- `_library_notes_` (73 of the 100 fields)
    assert notes_state_shim_attr("view") == "_library_notes_view"
    assert notes_state_shim_attr("sync_folder_text") == "_library_notes_sync_folder_text"
    # singular family -- `_library_note_` (21 fields)
    assert notes_state_shim_attr("preview") == "_library_note_preview"
    assert notes_state_shim_attr("editor_armed") == "_library_note_editor_armed"
    # Folder-Files family -- `_library_file_notes_` (5 fields), whose dataclass
    # names KEEP the `file_notes_` marker because Notes owns two reader
    # destinations and three names would otherwise collide.
    assert (
        notes_state_shim_attr("file_notes_workspace")
        == "_library_file_notes_workspace"
    )
    assert (
        notes_state_shim_attr("file_notes_reader_preferences")
        == "_library_file_notes_reader_preferences"
    )
    # bare-underscore family -- the one field with no "note" prefix word
    assert notes_state_shim_attr("selected_note_id") == "_selected_note_id"


@pytest.mark.unit
def test_the_two_reader_destinations_never_collapse_onto_one_field() -> None:
    """Notes owns TWO reader destinations; their fields must stay distinct.

    `_replace_library_reader_preference` dispatches `"notes"` and
    `"notes_files"` to two different screen attributes. Stripping the
    `file_notes_` marker (as every other family's mapping strips its own)
    would map both onto one dataclass field and silently make the Folder-Files
    pane inherit the Database-Notes pane's geometry. This pins the three
    colliding names apart at both ends of the mapping.
    """
    field_names = {f.name for f in dataclasses.fields(LibraryNotesState)}
    for stem in ("reader_preferences", "reader_layout", "reader_persistence_locks"):
        assert stem in field_names
        assert f"file_notes_{stem}" in field_names
        assert notes_state_shim_attr(stem) == f"_library_notes_{stem}"
        assert (
            notes_state_shim_attr(f"file_notes_{stem}")
            == f"_library_file_notes_{stem}"
        )


@pytest.mark.unit
def test_notes_state_field_prefix_sets_are_real_state_fields() -> None:
    """Guards the single-source prefix mapping against drift: every name the
    three exception sets list must actually be a `LibraryNotesState` field (a
    typo or a stale entry here would otherwise silently shim nothing under the
    intended prefix and everything under the wrong one instead), and no name
    may appear in two of them (which would make the branch order, not the
    census, decide the flat name).
    """
    field_names = {f.name for f in dataclasses.fields(LibraryNotesState)}
    for label, names in (
        ("NOTE_SINGULAR_STATE_FIELDS", NOTE_SINGULAR_STATE_FIELDS),
        ("FILE_NOTES_STATE_FIELDS", FILE_NOTES_STATE_FIELDS),
        ("NOTE_UNPREFIXED_STATE_FIELDS", NOTE_UNPREFIXED_STATE_FIELDS),
    ):
        unknown = names - field_names
        assert not unknown, f"{label} names unknown fields: {unknown!r}"
    assert not (NOTE_SINGULAR_STATE_FIELDS & FILE_NOTES_STATE_FIELDS)
    assert not (NOTE_SINGULAR_STATE_FIELDS & NOTE_UNPREFIXED_STATE_FIELDS)
    assert not (FILE_NOTES_STATE_FIELDS & NOTE_UNPREFIXED_STATE_FIELDS)


@pytest.mark.unit
def test_wiring_and_blocked_attributes_stay_off_the_state_object() -> None:
    """The 3 prior-extracted coordinator attributes are WIRING and the 2
    shared focus attributes are BLOCKED shell state -- none may be a
    `LibraryNotesState` field, and all five must remain plain (non-property)
    attributes on `LibraryScreen`.
    """
    field_names = {f.name for f in dataclasses.fields(LibraryNotesState)}
    assert "import_controller" not in field_names
    assert "sync_controller" not in field_names
    assert "session" not in field_names
    assert "programmatic_focus_target" not in field_names
    assert "restoring_focus" not in field_names

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    shimmed = [
        attr
        for attr in (*_NOTES_WIRING_SCREEN_ATTRS, *_NOTES_BLOCKED_SCREEN_ATTRS)
        if isinstance(getattr(LibraryScreen, attr, None), property)
    ]
    assert not shimmed, f"wiring/blocked attributes were shimmed as state: {shimmed!r}"


@pytest.mark.unit
def test_the_four_member_list_entry_focus_family_stays_screen_owned() -> None:
    """The shell family the wave-8 plan flagged has NO notes member.

    Re-derived at this commit: `_arm_library_list_entry_focus` and
    `_disarm_library_list_entry_focus` assign and clear these four together,
    and only the media one is subsystem-named. None contains "note", so none
    was ever a notes candidate -- and none may become a `LibraryNotesState`
    field by some later widening of the census's substring.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    family = (
        "_library_pending_list_entry_focus",
        "_library_pending_list_entry_focus_anchor",
        "_library_pending_list_entry_media_return",
        "_library_list_entry_focus_generation",
    )
    shim_names = {
        notes_state_shim_attr(f.name) for f in dataclasses.fields(LibraryNotesState)
    }
    assert not (set(family) & shim_names), (
        "a shell list-entry-focus family member was shimmed as notes state"
    )
    shimmed = [
        attr
        for attr in family
        if isinstance(getattr(LibraryScreen, attr, None), property)
    ]
    assert not shimmed, f"shell family members were shimmed: {shimmed!r}"


#: Filled in by this series' own cleanup task (task 3, notes series 3/3): the
#: moved names whose screen delegator had ZERO references outside its own body
#: anywhere in the repo, across all SIX census spellings (attribute, bare
#: quoted string, kwarg, bare ``ast.Name``, unbound ``LibraryScreen.<name>``
#: attribute, and the runtime f-string), over ``tldw_chatbook/`` + every
#: ``Tests/`` root + ``Docs/`` + ``backlog/`` + ``scripts/`` +
#: ``Helper_Scripts/``, and then re-checked one by one with a broad ``git
#: grep`` over EVERY file type. A ``def`` is never counted as a caller (recipe
#: SS4's ``on_<Message>`` incident). The 74 whitelist-exempt names (70 ``@on``
#: + 4 ``action_*``) are exempt from the census outright; SS4's third member,
#: ``on_<message>`` NAME dispatch, contributes ZERO for notes and
#: ``test_no_notes_handler_is_name_dispatched_by_textual`` keeps that proven.
#: Original 26 plus TASK-31932 steps 68/70/73: 33 of 185 = 17.84%.
_NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset(
    (
        "_remember_library_notes_authority_focus",
        "_evacuate_library_notes_authority_focus",
        "_compact_library_notes_stage",
        "_restore_library_notes_settled_focus",
        "_library_notes_work_first_preferences",
        "_library_notes_role_target",
        "_library_notes_work_session_reader_width",
        "_apply_library_note_saved_presentation",
        "_apply_library_notes_operation_state",
        "_defer_library_notes_settled_focus_restore",
        "_exit_library_notes_lasting_sync",
        "_focus_library_note_import_control",
        "_focus_library_notes_lasting_control",
        "_install_library_notes_scroll_observers",
        "_library_note_meta_base_line",
        "_library_note_session_is_unsafe",
        "_library_note_status_line",
        "_library_notes_authority_root",
        "_library_notes_fallback_focus_target",
        "_library_notes_operation_is_current_and_active",
        "_move_library_notes_operation",
        "_note_word_count",
        "_queue_library_notes_scroll_interaction",
        "_record_library_notes_presented_focus",
        "_record_library_notes_scroll_interaction",
        "_remember_library_notes_responsive_focus",
        "_resolve_library_note_conflict",
        "_restore_library_notes_final_scroll",
        "_restore_library_notes_scroll_after_layout",
        "_restore_library_notes_scroll_offset",
        "_run_library_note_import_check",
        "_selected_library_note_handoff_payload",
        "_settle_library_notes_final_scroll",
    )
)


#: Every method Task 2 moved into ``LibraryNotesController``, under its
#: original ``LibraryScreen`` name. Derived from a full ``ast`` census of every
#: ``LibraryScreen`` class-body method whose name contains "note"
#: (case-insensitive): **291 raw ``FunctionDef`` matches, 285 unique names**
#: (the 6-name gap is a byte-identical DUPLICATE block dev shipped twice; see
#: the controller docstring) -- minus 100 exclusions: 57 unbound-fake-self, 11
#: further in-file ``LibraryScreen.<name>(self, ...)`` targets, 8 further
#: callers of that same shape, 8 further instance-attribute-monkeypatch, 5
#: not-notes-owned, 4 further members of the ``_library_note_session``
#: projection-property family, 3 shared-shell-helper, 1 further
#: class-monkeypatch, 1 module-globals-coupling, 1 test-bound-and-captured
#: (``_exit_library_note_editor_guarded``, held by SS3's conservative opening
#: rule -- NOT the media series' Form E, see the controller docstring's own
#: correction), and 1
#: ``partial(LibraryScreen.<name>, self, ...)`` target the first census's
#: direct-call-argument shape could not see -- found by this task's own
#: BATTERY, per recipe SS3's amended-RED-tuple rule.
#: NOT a prefix/substring shortcut. See ``library_notes_controller.py``'s
#: module docstring for the full per-name reasoning behind every exclusion.
_NOTES_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_activate_database_note_work_session",
    "_append_library_note_source_record",
    "_apply_library_note_presentation_state",
    "_apply_library_note_save_outcome",
    "_apply_library_note_saved_presentation",
    "_apply_library_notes_footer_context",
    "_apply_library_notes_operation_state",
    "_apply_library_notes_stage_legs",
    "_begin_library_note_load",
    "_begin_library_notes_operation",
    "_build_library_note_import_executor",
    "_capture_library_notes_recompose_state",
    "_commit_library_note_widgets_before_recompose",
    "_compact_library_notes_stage",
    "_defer_library_notes_settled_focus_restore",
    "_delete_library_note",
    "_discard_new_library_note_claimed",
    "_dispatch_database_note_identity_cleared",
    "_dispatch_library_notes_work_session",
    "_evacuate_library_notes_authority_focus",
    "_exit_library_notes_lasting_sync",
    "_export_library_note",
    "_finish_library_note_create",
    "_finish_library_notes_operation",
    "_fire_library_note_autosave",
    "_focus_library_note_conflict_callout",
    "_focus_library_note_control",
    "_focus_library_note_import_control",
    "_focus_library_note_validation_field",
    "_focus_library_notes_filter_input",
    "_focus_library_notes_lasting_control",
    "_gc_pending_blank_note",
    "_handle_file_notes_editable_opened",
    "_handle_file_notes_identity_cleared",
    "_handle_file_notes_reload_confirmation_changed",
    "_handle_file_notes_root_changed",
    "_install_library_notes_scroll_observers",
    "_invalidate_library_note_autosave",
    "_library_note_editor_active",
    "_library_note_editor_state",
    "_library_note_import_database",
    "_library_note_import_execution_active",
    "_library_note_import_folder_repository",
    "_library_note_meta_base_line",
    "_library_note_presentation_state",
    "_library_note_session_is_unsafe",
    "_library_note_status_line",
    "_library_note_work_pane_kwargs",
    "_library_notes_active_region",
    "_library_notes_authority_root",
    "_library_notes_canvas_kwargs",
    "_library_notes_fallback_focus_target",
    "_library_notes_focus_region",
    "_library_notes_focus_stage",
    "_library_notes_folder_target_options",
    "_library_notes_list_canvas_kwargs",
    "_library_notes_mutation_fenced",
    "_library_notes_operation_for_active_region",
    "_library_notes_operation_is_current_and_active",
    "_library_notes_role_target",
    "_library_notes_scroll_owner",
    "_library_notes_user_id",
    "_library_notes_widget_is_within",
    "_library_notes_work_first_preferences",
    "_library_notes_work_session_reader_width",
    "_mark_library_notes_user_interaction",
    "_mirror_library_file_notes_reader_preference",
    "_mirror_library_notes_reader_preference",
    "_move_library_notes_operation",
    "_note_word_count",
    "_notes_true_count_or_none",
    "_notify_library_note_create_warning",
    "_notify_library_note_delete_warning",
    "_notify_library_note_import_failure",
    "_notify_library_note_missing_warning",
    "_open_selected_library_note_handoff",
    "_project_library_note_entry_result",
    "_publish_library_note_import_snapshot",
    "_publish_library_notes_lasting_sync_snapshot",
    "_queue_library_notes_scroll_interaction",
    "_queue_library_notes_settled_focus_restore",
    "_read_library_note_editor_fields",
    "_reconcile_library_notes_list_canvas",
    "_record_library_notes_focus_interaction",
    "_record_library_notes_presented_focus",
    "_record_library_notes_scroll_interaction",
    "_refresh_after_library_note_import",
    "_rehydrate_library_notes_after_recompose",
    "_release_library_notes_focus_after_snapshot",
    "_remember_library_notes_authority_focus",
    "_remember_library_notes_responsive_focus",
    "_remove_library_note_source_record",
    "_reset_library_note_editor_state",
    "_resolve_library_note_conflict",
    "_restore_library_note_delete_origin",
    "_restore_library_notes_after_targeted_sync",
    "_restore_library_notes_authority_focus",
    "_restore_library_notes_final_scroll",
    "_restore_library_notes_focus_identity",
    "_restore_library_notes_scroll_after_layout",
    "_restore_library_notes_scroll_offset",
    "_restore_library_notes_settled_focus",
    "_return_from_library_notes_task",
    "_route_library_note_validation_field",
    "_run_library_note_import_check",
    "_selected_library_note_handoff_payload",
    "_selected_library_notes_tree_row",
    "_set_library_notes_source",
    "_settle_library_notes_final_scroll",
    "_show_library_database_notes",
    "_show_library_file_notes",
    "_show_library_note_shortcut_refusal",
    "_sync_library_file_notes_reader_layout_from_shell",
    "_sync_library_notes_source_controls",
    "_try_switch_retained_library_notes_route",
    "_undo_library_note_delete",
    "_update_library_note_meta_static",
    "_update_library_notes_responsive_state",
    "action_library_note_editor_back",
    "action_library_notes_escape",
    "action_library_notes_focus_filter",
    "action_library_notes_new",
    "handle_library_note_back",
    "handle_library_note_body_changed",
    "handle_library_note_conflict_overwrite",
    "handle_library_note_conflict_reload",
    "handle_library_note_context_back",
    "handle_library_note_context_keywords_changed",
    "handle_library_note_context_open",
    "handle_library_note_copy",
    "handle_library_note_delete",
    "handle_library_note_delete_cancel",
    "handle_library_note_delete_receipt_dismiss",
    "handle_library_note_discard_new",
    "handle_library_note_edit_mode",
    "handle_library_note_export_markdown",
    "handle_library_note_export_text",
    "handle_library_note_import_add_source",
    "handle_library_note_import_cancel",
    "handle_library_note_import_check",
    "handle_library_note_import_collision_choice",
    "handle_library_note_import_collision_name",
    "handle_library_note_import_confirm_match",
    "handle_library_note_import_destination",
    "handle_library_note_import_item_action",
    "handle_library_note_import_item_choice",
    "handle_library_note_import_page",
    "handle_library_note_import_retry",
    "handle_library_note_keywords_changed",
    "handle_library_note_load_retry",
    "handle_library_note_preview_toggle",
    "handle_library_note_save",
    "handle_library_note_title_changed",
    "handle_library_note_use_in_console",
    "handle_library_note_work_pane_editor_ready",
    "handle_library_notes_add_from_files",
    "handle_library_notes_create_back",
    "handle_library_notes_create_blank",
    "handle_library_notes_create_template",
    "handle_library_notes_export",
    "handle_library_notes_folder_row",
    "handle_library_notes_import_back",
    "handle_library_notes_lasting_activate",
    "handle_library_notes_lasting_apply",
    "handle_library_notes_lasting_back",
    "handle_library_notes_lasting_check",
    "handle_library_notes_lasting_choice",
    "handle_library_notes_lasting_comparison",
    "handle_library_notes_lasting_comparison_return",
    "handle_library_notes_lasting_dismiss",
    "handle_library_notes_lasting_folder_requested",
    "handle_library_notes_lasting_history",
    "handle_library_notes_lasting_history_page",
    "handle_library_notes_lasting_history_return",
    "handle_library_notes_lasting_review_page",
    "handle_library_notes_lasting_root_action",
    "handle_library_notes_lasting_root_page",
    "handle_library_notes_lasting_setup_changed",
    "handle_library_notes_lasting_undo",
    "handle_library_notes_new",
    "handle_library_notes_relationship_choice",
    "handle_library_notes_select_clear",
    "handle_library_notes_select_toggle",
    "handle_library_notes_sort",
    "handle_library_notes_sort_choice",
)

_NOTES_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_build_library_note_import_executor",
        "_library_notes_widget_is_within",
        "_note_word_count",
    }
)

_NOTES_CONTROLLER_BOUND_NAMES: tuple[str, ...] = (
    # -- framework services, live-read from the screen on every access (16)
    "_footer_shortcut_registration",
    "app",
    "app_instance",
    "call_after_refresh",
    "call_later",
    "focus_chain",
    "focused",
    "is_mounted",
    "is_running",
    "query",
    "query_one",
    "refresh",
    "register_footer_shortcuts",
    "run_worker",
    "set_focus",
    "watch",
    # -- shared shell state this cluster READS (getter-only accessors) (10)
    "_library_canvas_projection_depth",
    "_library_lifecycle",
    "_library_onboarding_all_empty",
    "_library_rail_collapsed",
    "_library_snapshot_state_generation",
    "_local_source_counts",
    "_local_source_records",
    "_media_state",
    "_prompts_state",
    "_rag_search_state",
    # -- shared shell state this cluster also WRITES (getter + setter) (6)
    "_library_canvas_resync_pending",
    "_library_navigation_context_generation",
    "_library_notes_programmatic_focus_target",
    "_library_notes_restoring_focus",
    "_library_selected_row_id",
    "_pending_library_source_open",
    # -- the 3 prior-extracted notes WIRING coordinator instances (3)
    "_library_note_import_controller",
    "_library_note_session",
    "_library_notes_sync_controller",
    # -- screen-resident methods a moved body still calls (named late-binding callables) (69)
    "_acknowledge_library_destination_change",
    "_active_library_rail",
    "_advance_library_stage_interaction",
    "_apply_library_emergency_geometry",
    "_apply_library_notes_stage_visibility",
    "_apply_library_notes_stage_visibility_for_resize",
    "_arm_library_list_entry_focus",
    "_arm_library_note_editor",
    "_begin_library_note_create",
    "_build_library_notes_state",
    "_build_library_notes_tree_projection",
    "_build_library_shell_input",
    "_capture_library_notes_browse_return_receipt",
    "_capture_library_notes_focus_identity",
    "_compose_library_rail_top_action",
    "_compose_workspaces_rail_body",
    "_create_library_note",
    "_delete_library_note_claimed",
    "_discard_new_library_note",
    "_exit_library_note_editor_guarded",
    "_file_notes_active",
    "_flush_library_note_save",
    "_focus_library_control",
    "_hide_library_adaptive_reader_rail_collapse",
    "_invalidate_library_workspace_depth_state",
    "_library_compose_scoped_ref",
    "_library_entry_route_key",
    "_library_header_line",
    "_library_layout_ref",
    "_library_note_dirty",
    "_library_note_keywords_from_input",
    "_library_note_template_fields",
    "_library_note_version",
    "_library_notes_compact_stage_applies",
    "_library_notes_compact_workflow_active",
    "_library_notes_focused_task_active",
    "_library_notes_footer_shortcuts",
    "_library_notes_restore_guard_is_current",
    "_library_notes_workflow_active",
    "_library_rail_preferences",
    "_library_rail_search_placeholder",
    "_library_workspace_depth_state",
    "_locate_library_notes_tree_target",
    "_open_library_export_canvas",
    "_patch_library_note_list_from_session",
    "_project_library_media_stage_classes",
    "_push_library_note_import_picker",
    "_reconcile_library_notes_tree_mutation",
    "_refresh_library_note_detail",
    "_refresh_local_source_snapshot",
    "_register_footer_shortcuts",
    "_replace_library_canvas_child",
    "_request_library_notes_tree_initial_load",
    "_request_library_notes_tree_slice",
    "_return_to_library_database_notes",
    "_run_library_note_import_execution",
    "_run_library_service_call",
    "_save_library_note",
    "_schedule_library_note_autosave",
    "_select_library_rail_row",
    "_set_library_destination_with_conversation_fence",
    "_source_record_id",
    "_supersede_library_notes_navigation",
    "_sync_library_ingest_rail_for_width",
    "_sync_library_notes_reader_layout_from_shell",
    "_sync_library_ordinary_rail_width_contract",
    "_transition_library_notes_presentation",
    "_write_library_note_export_file",
)


@pytest.mark.unit
def test_notes_cluster_method_names_are_genuinely_notes_named() -> None:
    """Guards the hand-kept cluster list against drift with the census:
    every name must contain "note" (case-insensitive) -- a typo here would
    silently test the wrong surface.
    """
    not_notes_named = [
        n for n in _NOTES_CLUSTER_METHOD_NAMES if "note" not in n.lower()
    ]
    assert not not_notes_named, f"non-notes-named cluster entries: {not_notes_named!r}"
    assert len(_NOTES_CLUSTER_METHOD_NAMES) == 185, (
        f"expected 185 moved names, got {len(_NOTES_CLUSTER_METHOD_NAMES)}"
    )
    assert len(set(_NOTES_CLUSTER_METHOD_NAMES)) == len(_NOTES_CLUSTER_METHOD_NAMES), (
        "duplicate entries in _NOTES_CLUSTER_METHOD_NAMES"
    )


@pytest.mark.unit
def test_notes_controller_owns_its_cluster() -> None:
    """Every one of the 185 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    ``test_media_controller_owns_its_cluster``.
    """
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )

    missing = [
        name
        for name in _NOTES_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryNotesController, name, None))
    ]
    assert not missing, f"LibraryNotesController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_notes_handlers() -> None:
    """Every one of the 185 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 3 staticmethods,
    to the module-level controller CLASS) -- unless a later cleanup task pruned
    it.

    A same-name forwarding check, not a loose "the controller is referenced
    somewhere" substring check. Skips the names in
    ``_NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED`` (task 3's census) and instead
    asserts each such name is genuinely ABSENT from ``LibraryScreen``, so a
    future accidental re-add fails loudly here rather than silently
    reintroducing dead code.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _NOTES_CLUSTER_METHOD_NAMES:
        if name in _NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen but is back -- either "
                "wire it as a delegator again or drop it from "
                "_NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(rf"_notes_controller\.{escaped}\(", src) and not re.search(
            rf"LibraryNotesController\.{escaped}\(", src
        ):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_notes_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 3 staticmethod names in the cluster forward to the CLASS.

    A ``@staticmethod`` has no ``self`` to reach ``self._notes_controller``
    through, so its delegator names the module-level controller class directly
    -- the conversations exemplar's own corrected shape (recipe SS11, "the
    static-method delegator pattern"), reused unchanged by every series since.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert _NOTES_CLUSTER_STATICMETHOD_NAMES <= set(_NOTES_CLUSTER_METHOD_NAMES), (
        "a staticmethod name is not in the mover tuple"
    )
    not_class_forwarding = []
    for name in _NOTES_CLUSTER_STATICMETHOD_NAMES:
        if name in _NOTES_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen but is back"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_class_forwarding.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        if not re.search(rf"LibraryNotesController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_notes_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    The moved bodies spell the ORIGINAL flat names, so each one has to keep
    resolving on the controller -- through the same single-source
    ``notes_state_shim_attr()`` the screen's own task-1 shim block uses.
    """
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryNotesState)}
    assert field_names, "state object is empty"
    missing = []
    for name in sorted(field_names):
        shim_attr = notes_state_shim_attr(name)
        if not isinstance(getattr(LibraryNotesController, shim_attr, None), property):
            missing.append(shim_attr)
    assert not missing, (
        f"no notes controller shim property found for state field(s): {missing!r}"
    )


@pytest.mark.unit
def test_every_controller_shim_reads_and_writes_its_own_state_field() -> None:
    """Each generated controller property is a real two-way shim, not a stub.

    The same closure-binding trap ``test_every_shim_reads_and_writes_its_own_
    state_field`` rules out for the SCREEN's block, aimed at the controller's
    own permanent one: a `for` loop over `dataclasses.fields` gives every
    generated property the LAST field unless the name is bound as a default
    argument, which would satisfy a bare `isinstance(..., property)` check
    while silently aliasing 100 names onto one field.
    """
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )

    controller = object.__new__(LibraryNotesController)
    state = LibraryNotesState()
    controller._notes_state_accessor = lambda: state

    field_names = sorted(f.name for f in dataclasses.fields(LibraryNotesState))
    read_mismatch = []
    write_mismatch = []
    for name in field_names:
        shim_attr = notes_state_shim_attr(name)
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
    written = [getattr(state, name) for name in field_names]
    assert len({id(value) for value in written}) == len(field_names)


@pytest.mark.unit
def test_notes_controller_binds_every_name_its_moved_bodies_use() -> None:
    """Constructor-binding coverage: the byte-for-byte canon's own contract.

    A moved body is never edited, so every non-state name it spells as
    ``self.<name>`` (or reaches by ``getattr(self, "<literal>")``, or hands to
    the shared ``_sync_library_canvas`` dispatcher as bare ``self``) has to
    keep resolving -- on the CONTROLLER now, not the screen. This asserts the
    resolution exists at the CLASS level rather than on a constructed
    instance, deliberately: framework forwards raise off the app tree, so an
    instance probe would report a false failure for a binding that is in fact
    present and correct.

THREE of these names appear in NO moved body at all and would be missed by
    that walk -- a re-derivation over the moved bodies names exactly these
    three and nothing else: ``_library_canvas_projection_depth`` (read as
    ``getattr(screen, "_library_canvas_projection_depth", 0)``),
    ``_library_canvas_resync_pending`` (assigned) and ``is_running`` (read at
    ``canvas_sync.py:535``). All three are reached by the SHARED
    ``_sync_library_canvas`` dispatcher through the bare ``self`` 26 movers
    forward into it -- the unbound-attribute-escape shape one indirection
    further out than the skills series' own ``focused``. The conversations and
    media controllers each bind the first two for the identical reason; the
    third is this cluster's own. (An earlier draft of this paragraph said TWO
    and omitted ``is_running`` -- corrected in review, after measuring rather
    than re-reading.)
    """
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )

    assert len(_NOTES_CONTROLLER_BOUND_NAMES) == 103, (
        f"expected 103 bound names, got {len(_NOTES_CONTROLLER_BOUND_NAMES)}"
    )
    assert len(set(_NOTES_CONTROLLER_BOUND_NAMES)) == len(
        _NOTES_CONTROLLER_BOUND_NAMES
    ), "duplicate entries in _NOTES_CONTROLLER_BOUND_NAMES"
    unbound = [
        name
        for name in _NOTES_CONTROLLER_BOUND_NAMES
        if not isinstance(getattr(LibraryNotesController, name, None), property)
    ]
    assert not unbound, (
        "moved bodies reference these names, but the controller binds no "
        f"property for them: {unbound!r}"
    )


@pytest.mark.unit
def test_no_notes_handler_is_name_dispatched_by_textual() -> None:
    """Recipe SS4's THIRD delegator-prune whitelist member, RESOLVED for notes.

    The wave-8 plan predicted notes would own ``on_<message>`` NAME-dispatched
    handlers (media's were zero, so the member has been untested since
    prompts). It does not, and this pins the answer against Textual's own
    dispatch rather than against a reference census -- the exact check recipe
    SS4 prescribes: "construct the ``Message`` subclass it would receive and
    confirm ``handler_name`` matches, or confirm no ``Message`` in the
    subsystem's own widgets produces that string."

    Both directions are asserted. Notes' 35 Message-typed ``@on`` handlers are
    DECORATOR-dispatched (whitelist member 1); none is named the
    ``handler_name`` its own Message computes. And no ``Message`` anywhere in
    ``Widgets/Library`` computes a ``handler_name`` that a notes-named
    ``LibraryScreen`` method answers to.
    """
    import importlib
    import pkgutil

    from textual.message import Message

    import tldw_chatbook.Widgets.Library as widgets_library
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    handler_names: dict[str, list[str]] = {}
    for info in pkgutil.iter_modules(widgets_library.__path__):
        try:
            module = importlib.import_module(
                f"tldw_chatbook.Widgets.Library.{info.name}"
            )
        except Exception:  # pragma: no cover - optional widget deps
            continue
        for attr in dir(module):
            value = getattr(module, attr, None)
            if (
                isinstance(value, type)
                and issubclass(value, Message)
                and value is not Message
            ):
                handler_names.setdefault(value.handler_name, []).append(
                    f"{info.name}.{attr}"
                )
            if isinstance(value, type):
                for inner in vars(value).values():
                    if (
                        isinstance(inner, type)
                        and issubclass(inner, Message)
                        and inner is not Message
                    ):
                        handler_names.setdefault(inner.handler_name, []).append(
                            f"{info.name}.{attr}.{inner.__name__}"
                        )
    assert handler_names, "no Library Message classes found -- census is vacuous"

    # Direction 1: no notes-named cluster method is itself a handler_name.
    name_dispatched = sorted(set(_NOTES_CLUSTER_METHOD_NAMES) & set(handler_names))
    assert not name_dispatched, (
        "these moved names ARE Textual name-dispatch targets and must be kept "
        f"as screen delegators unconditionally: {name_dispatched!r}"
    )

    # Direction 2: every LibraryScreen method that IS a handler_name is
    # accounted for, and none of them is notes-owned.
    screen_dispatched = sorted(
        name
        for name in handler_names
        if callable(getattr(LibraryScreen, name, None))
        and "note" in name.lower()
    )
    assert not screen_dispatched, (
        "LibraryScreen carries notes-owned name-dispatched handlers the "
        f"cluster census never enumerated: {screen_dispatched!r}"
    )
