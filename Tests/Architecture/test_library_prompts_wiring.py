"""Prompts extraction series: state object and controller are screen-wired.

Wave-6 Task 1 (prompts series 1/3, state PR) and Task 2 (prompts series
2/3, controller PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; skills series precedent: ``Tests/
Architecture/test_library_skills_wiring.py``, its state-PR-era shape -- the
closest match, since Prompts is likewise a THREE-prefix subsystem). Every
field ``LibraryPromptsState`` declares must have a matching generated
property shim on ``LibraryScreen``, resolved via ``prompt_state_shim_attr()``
-- the single-source three-way prefix mapping (``_library_prompt_`` default,
``_library_prompts_`` for the plural-named subset, bare ``_`` for the one
unprefixed field, ``selected_prompt_id``) documented in
``library_prompts_state.py``'s own module docstring.

**What the per-field sweep can and cannot prove.** Because the screen's shim
loop and this file's sweep both call ``prompt_state_shim_attr()``, the sweep
proves the two are CONSISTENT -- it cannot prove the mapping is CORRECT. A
review pass demonstrated this concretely: deleting the plural branch from
``prompt_state_shim_attr()`` outright leaves the sweep at 5 passed, because
screen and test then agree on the same wrong answer. ``test_prompt_state_
shim_attr_maps_each_prefix_family_to_its_literal_name`` below closes that
hole with hard-coded expected strings -- one per prefix family, including the
bare-underscore one -- which is the only assertion here that would fail on
such a mutation. The sweep still earns its place for the other half of the
job: that each shim genuinely reads AND writes its own state field rather
than merely existing as a property.

The state module under test is ``tldw_chatbook.UI.Library_Modules.library_
prompts_state`` -- NOT the unrelated, pre-existing ``tldw_chatbook.Library.
library_prompts_state`` domain module of the same basename (see that
docstring's own note on the collision).

**Task 2 (controller PR)** adds the full-cluster ownership / same-name-
delegator-forwarding / staticmethod-class-forwarding / controller-state-shim
checks (``_PROMPTS_CLUSTER_METHOD_NAMES``, 139 names) plus a
constructor-binding coverage check (``_PROMPTS_CONTROLLER_BOUND_NAMES``, 42
names) that no prior series' wiring test carries -- added here because the
skills series shipped a silent production regression precisely in that gap
(a moved body's ``getattr(self, "focused", None)`` with no ``focused``
property bound; recipe §3's unbound-attribute-escape entry). See
``library_prompts_controller.py``'s own module docstring for the full
161-candidate derivation and the 22 exclusions.
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_prompts_state import (
    PROMPT_UNPREFIXED_STATE_FIELDS,
    PROMPTS_PLURAL_STATE_FIELDS,
    LibraryPromptsState,
    prompt_state_shim_attr,
)

#: The ownership census's own MOVE count (recipe §2 script, substring
#: "prompt" over every ``__init__``-stored attribute of ``LibraryScreen``):
#: 46 fields found, 3 WIRING (live controller instances -- see below), 0
#: BLOCKED, so 43 move. Pinned here so a field silently added to or dropped
#: from the dataclass fails loudly instead of quietly shrinking the shim
#: surface this file checks.
_EXPECTED_PROMPT_STATE_FIELD_COUNT = 43

#: The 3 WIRING fields the state PR deliberately left on ``LibraryScreen``
#: (the ``_conversation_reader_controller``/``_library_collections_capture_
#: controller``/``_library_skill_import_coordinator`` precedent): each holds
#: a live controller instance, not data.
_PROMPT_WIRING_SCREEN_ATTRS: tuple[str, ...] = (
    "_library_prompt_history_controller",
    "_library_prompt_browse_controller",
    "_library_prompt_collections_controller",
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryPromptsState)}
    assert field_names, "state object is empty"
    assert len(field_names) == _EXPECTED_PROMPT_STATE_FIELD_COUNT, (
        f"expected {_EXPECTED_PROMPT_STATE_FIELD_COUNT} prompt fields, "
        f"got {len(field_names)}"
    )
    missing = []
    for name in sorted(field_names):
        shim_attr = prompt_state_shim_attr(name)
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
    property)` check while silently aliasing 43 names onto one field. This
    round-trips a distinct sentinel through every name to rule that out.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    screen = object.__new__(LibraryScreen)
    state = LibraryPromptsState()
    screen._prompts_state = state

    field_names = sorted(f.name for f in dataclasses.fields(LibraryPromptsState))
    read_mismatch = []
    write_mismatch = []
    for name in field_names:
        shim_attr = prompt_state_shim_attr(name)
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
def test_prompt_state_shim_attr_maps_each_prefix_family_to_its_literal_name() -> None:
    """The three-way mapping, pinned against LITERAL expected strings.

    The only assertion in this file that is not self-referential: every
    other check resolves the expected name by calling the same
    `prompt_state_shim_attr()` the screen's own shim loop calls, so screen
    and test agree even when the mapping is wrong. A review pass proved
    that hole is real -- deleting the plural branch from
    `prompt_state_shim_attr()` leaves the rest of this file at 5 passed.
    One field per prefix family, spelled out:
    """
    # plural family -- `_library_prompts_` (the list/browse/import surface)
    assert prompt_state_shim_attr("view") == "_library_prompts_view"
    # singular family -- `_library_prompt_` (the cluster default)
    assert prompt_state_shim_attr("dirty") == "_library_prompt_dirty"
    # bare-underscore family -- the one field with no prompt(s) prefix word
    assert prompt_state_shim_attr("selected_prompt_id") == "_selected_prompt_id"


@pytest.mark.unit
def test_prompt_state_field_prefix_sets_are_real_state_fields() -> None:
    """Guards the single-source prefix mapping against drift: every name
    either exception set lists must actually be a `LibraryPromptsState`
    field (a typo or a stale entry here would otherwise silently shim
    nothing under the intended prefix and everything under the wrong one
    instead).
    """
    field_names = {f.name for f in dataclasses.fields(LibraryPromptsState)}
    unknown_plural = PROMPTS_PLURAL_STATE_FIELDS - field_names
    unknown_bare = PROMPT_UNPREFIXED_STATE_FIELDS - field_names
    assert not unknown_plural, (
        f"PROMPTS_PLURAL_STATE_FIELDS names unknown fields: {unknown_plural!r}"
    )
    assert not unknown_bare, (
        f"PROMPT_UNPREFIXED_STATE_FIELDS names unknown fields: {unknown_bare!r}"
    )


@pytest.mark.unit
def test_prompt_state_prefix_sets_do_not_overlap() -> None:
    """A field can only be in at most one of the two exception sets --
    membership in both would make `prompt_state_shim_attr`'s branch order
    silently pick one and hide the other's intent."""
    overlap = PROMPTS_PLURAL_STATE_FIELDS & PROMPT_UNPREFIXED_STATE_FIELDS
    assert not overlap, f"fields claimed by both prefix exception sets: {overlap!r}"


@pytest.mark.unit
def test_wiring_fields_stay_off_the_state_object() -> None:
    """The 3 prompt controller attributes are WIRING, not state -- they must
    NOT be `LibraryPromptsState` fields, and must remain plain
    (non-property) attributes on `LibraryScreen`.
    """
    field_names = {f.name for f in dataclasses.fields(LibraryPromptsState)}
    assert "history_controller" not in field_names
    assert "browse_controller" not in field_names
    assert "collections_controller" not in field_names

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    shimmed = [
        attr
        for attr in _PROMPT_WIRING_SCREEN_ATTRS
        if isinstance(getattr(LibraryScreen, attr, None), property)
    ]
    assert not shimmed, f"wiring attributes were shimmed as state: {shimmed!r}"


#: Every method Task 2 moved into `LibraryPromptsController`, under its
#: original `LibraryScreen` name. Derived from a full `ast` census of every
#: `LibraryScreen` class-body method whose name contains "prompt"
#: (case-insensitive): **161 raw `FunctionDef` matches, 161 unique names**
#: (no property/setter-pair gap, unlike Skills' own 133/127) -- minus 22
#: exclusions: 2 screen-identity (recipe §3's sixth bypass shape, Form C:
#: an inlined `self.app.screen is not self`), 14 unbound-fake-self (3 of
#: them reached through an INDIRECTION a bare `LibraryScreen.<name>(` grep
#: cannot see -- a fake-harness CLASS ATTRIBUTE, a `parametrize` tuple of
#: unbound functions, and a `getattr(LibraryScreen, "<name>")` string-name
#: dispatch), 3 instance-attribute-monkeypatch, 2 module-globals-coupling,
#: and 1 merely-delegate-to-existing-controller `@property`. NOT a prefix/
#: substring shortcut. See `library_prompts_controller.py`'s module
#: docstring for the full per-name reasoning behind every exclusion.
_PROMPTS_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_adopt_library_prompt_persisted_detail",
    "_apply_library_prompt_collection",
    "_apply_library_prompt_detail_failure",
    "_apply_library_prompt_save_outcome",
    "_apply_library_prompt_working_copy",
    "_arm_library_prompt_editor",
    "_await_library_prompt_durable_call",
    "_await_library_prompt_save_call",
    "_capture_library_prompt_block_state",
    "_capture_library_prompts_filter_cursor",
    "_claim_library_prompt_detail_generation",
    "_clear_library_prompt_delete_pending",
    "_clear_library_prompt_selection",
    "_confirm_library_prompt_history_restore",
    "_current_library_prompt_editor_state",
    "_delete_library_prompts",
    "_detach_library_prompt_working_copy",
    "_enter_library_prompt_conflict",
    "_enter_library_prompt_create_editor",
    "_exit_library_prompt_editor_guarded",
    "_export_library_prompt",
    "_flush_library_prompts_search",
    "_initialize_library_prompt_history",
    "_invalidate_library_prompt_detail_generation",
    "_invalidate_library_prompt_history",
    "_invalidate_library_prompts_browse",
    "_library_prompt_action_artifact_type",
    "_library_prompt_artifact_fields",
    "_library_prompt_basic_unavailable_reason",
    "_library_prompt_can_update_original",
    "_library_prompt_delete_fingerprint",
    "_library_prompt_detail_failure_notice",
    "_library_prompt_detail_request_is_current",
    "_library_prompt_editor_active",
    "_library_prompt_history_action_is_current",
    "_library_prompt_legacy_recipe_requires_conversion",
    "_library_prompt_loading_notice",
    "_library_prompt_markdown_artifact_fields",
    "_library_prompt_mutation_is_current",
    "_library_prompt_nearest_survivor_focus",
    "_library_prompt_text_fields_match_state",
    "_library_prompt_work_pane_kwargs",
    "_library_prompt_write_worker_is_active",
    "_library_prompts_canvas_kwargs",
    "_library_prompts_focus_identity",
    "_library_prompts_list_canvas_kwargs",
    "_load_library_prompt_memberships",
    "_mark_library_prompt_dirty",
    "_mirror_library_prompts_reader_preference",
    "_notify_library_prompt_delete_failure",
    "_notify_library_prompt_legacy_recipe_requires_conversion",
    "_notify_library_prompt_unrepresentable_markdown",
    "_notify_library_prompt_unsupported_artifact_type",
    "_notify_prompt_dirty_veto",
    "_on_library_prompt_history_region_ready",
    "_open_library_prompt_colliding_with_current_name",
    "_open_library_prompt_delete_confirmation",
    "_prompts_count_or_none",
    "_queue_library_prompts_search",
    "_read_library_prompt_editor_fields",
    "_reconcile_library_prompt_history_region",
    "_reconcile_library_prompt_memberships",
    "_refocus_library_prompt_delete_action",
    "_refresh_library_prompt_after_membership_apply",
    "_refresh_library_prompt_detail",
    "_request_library_prompt_history_count",
    "_request_library_prompt_history_page",
    "_resolve_library_prompt_conflict",
    "_resolve_library_prompt_create_conflict",
    "_restore_library_prompt_history",
    "_restore_library_prompts_focus",
    "_restore_library_prompts_scope",
    "_return_to_library_prompt_create_draft",
    "_save_library_prompt",
    "_set_library_prompt_discard_enabled",
    "_start_library_prompts_import",
    "_stop_library_prompts_search_debounce",
    "_sync_library_prompt_collection_label",
    "_sync_library_prompt_history_region",
    "_sync_library_prompt_memberships",
    "_sync_library_prompt_mutation_presentation",
    "_sync_library_prompt_open_existing_button",
    "_sync_library_prompt_save_action_widgets",
    "_sync_library_prompt_selection",
    "_sync_library_prompts_browse_result",
    "_sync_library_prompts_reader_layout_from_shell",
    "_undo_library_prompt_delete",
    "_update_library_prompt_meta_static",
    "_update_library_prompt_status_static",
    "action_library_prompt_editor_back",
    "handle_library_prompt_back",
    "handle_library_prompt_conflict_reload",
    "handle_library_prompt_conflict_save_new",
    "handle_library_prompt_convert",
    "handle_library_prompt_copy",
    "handle_library_prompt_delete",
    "handle_library_prompt_delete_receipt_dismiss",
    "handle_library_prompt_delete_undo",
    "handle_library_prompt_detail_retry",
    "handle_library_prompt_discard",
    "handle_library_prompt_duplicate",
    "handle_library_prompt_editor_mode",
    "handle_library_prompt_export",
    "handle_library_prompt_history_closed",
    "handle_library_prompt_history_opened",
    "handle_library_prompt_history_reload",
    "handle_library_prompt_history_request_page",
    "handle_library_prompt_history_restore",
    "handle_library_prompt_history_retry_count",
    "handle_library_prompt_history_row",
    "handle_library_prompt_input_changed",
    "handle_library_prompt_memberships_apply",
    "handle_library_prompt_memberships_manage",
    "handle_library_prompt_open_existing",
    "handle_library_prompt_recipe_starter_changed",
    "handle_library_prompt_save",
    "handle_library_prompt_textarea_changed",
    "handle_library_prompts_clear_selection",
    "handle_library_prompts_collection",
    "handle_library_prompts_delete_selected",
    "handle_library_prompts_export",
    "handle_library_prompts_export_selected",
    "handle_library_prompts_filter_changed",
    "handle_library_prompts_import",
    "handle_library_prompts_import_browse",
    "handle_library_prompts_import_cancel",
    "handle_library_prompts_import_path_changed",
    "handle_library_prompts_import_path_submitted",
    "handle_library_prompts_import_run",
    "handle_library_prompts_retry",
    "handle_library_prompts_select",
    "handle_library_prompts_select_page",
    "handle_library_prompts_selection_done",
    "on_prompt_block_editor_back_requested",
    "on_prompt_block_editor_block_action_requested",
    "on_prompt_block_editor_block_field_changed",
    "on_prompt_block_editor_save_as_prompt_requested",
    "on_prompt_block_editor_save_as_recipe_requested",
    "on_prompt_block_editor_update_original_requested",
)

#: The 1 name above that is a `@staticmethod` on `LibraryScreen`. Its
#: delegator forwards straight to the module-level `LibraryPromptsController`
#: CLASS (the skills/export/collections/search+RAG/ingest wiring tests'
#: "static-method delegator pattern" precedent), not through
#: `self._prompts_controller`.
_PROMPTS_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_restore_library_prompts_scope",
    }
)

#: Filled in by this series' own cleanup task (task 3, prompts series 3/3)
#: with the moved names whose screen delegator has zero external references.
#: Deliberately EMPTY here: a controller PR moves bodies and leaves a
#: delegator under every one of them, so the delegator-prune census has not
#: been run yet. The skip/absence-assertion pair below is wired now (rather
#: than added later) so task 3's own edit is a one-line frozenset change,
#: matching `_INGEST_CLUSTER_SCREEN_DELEGATOR_PRUNED`/
#: `_SKILLS_CLUSTER_SCREEN_DELEGATOR_PRUNED` in their post-cleanup shape.
_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED: frozenset[str] = frozenset()

#: Every name a moved body references that is NOT this controller's own
#: `LibraryPromptsState` field and NOT another mover -- i.e. the complete
#: constructor-binding surface, derived mechanically from an `ast` walk of
#: all 139 moved bodies (every `self.<attr>` load/store, plus every
#: `getattr(self, "<literal>")` -- the shape recipe §3's sixth-bypass entry
#: records as invisible to a plain `self.<attr>` census, and the one that
#: cost the skills series a silent production regression on `focused`).
#: Grouped exactly as the controller's own constructor groups them.
_PROMPTS_CONTROLLER_BOUND_NAMES: tuple[str, ...] = (
    # -- framework services, live-read from the screen on every access (12)
    "app",
    "app_instance",
    "call_after_refresh",
    "focused",
    "is_mounted",
    "is_running",
    "query",
    "query_one",
    "refresh",
    "run_worker",
    "set_timer",
    "workers",
    # -- general Library-wide shell helpers, named constructor callables (12)
    "_arm_library_list_entry_focus",
    "_focus_library_control",
    "_library_entry_reconcile_is_current",
    "_library_entry_route_key",
    "_library_list_canvas_showing_list",
    "_library_note_keywords_from_input",
    "_open_library_export_canvas",
    "_refresh_local_source_snapshot",
    "_run_library_service_call",
    "_safe_text",
    "_sanitize_media_field",
    "_sanitize_note_content",
    # -- shared shell state this cluster READS (getter-only accessors) (4)
    "_library_pending_list_entry_focus",
    "_library_selected_row_id",
    "_library_snapshot_state_generation",
    "_local_source_counts",
    # -- the 3 prior-extracted prompt WIRING controller instances (3)
    "_library_prompt_browse_controller",
    "_library_prompt_collections_controller",
    "_library_prompt_history_controller",
    # -- the one merely-delegate-to-existing-controller property (1)
    "_library_prompt_history_state",
    # -- named late-binding callables for the 10 excluded prompt methods
    #    a mover still calls internally (10)
    "_apply_library_prompts_import_status",
    "_build_library_prompts_state",
    "_flush_library_prompt_save",
    "_persist_library_prompt_editor_mode",
    "_request_library_prompts_browse",
    "_reset_library_prompt_editor_state",
    "_run_library_prompts_import",
    "_settle_library_prompt_delete",
    "_stage_library_prompt_for_console",
    "_write_library_prompt_export_file",
)


@pytest.mark.unit
def test_prompts_cluster_method_names_are_genuinely_prompt_named() -> None:
    """Guards the hand-kept cluster list against drift with the census: every
    name must contain "prompt" (case-insensitive) -- a typo here would
    silently test the wrong surface.
    """
    not_prompt_named = [
        n for n in _PROMPTS_CLUSTER_METHOD_NAMES if "prompt" not in n.lower()
    ]
    assert not not_prompt_named, (
        f"non-prompt-named cluster entries: {not_prompt_named!r}"
    )
    assert len(_PROMPTS_CLUSTER_METHOD_NAMES) == 139, (
        f"expected 139 moved names, got {len(_PROMPTS_CLUSTER_METHOD_NAMES)}"
    )
    assert len(set(_PROMPTS_CLUSTER_METHOD_NAMES)) == len(
        _PROMPTS_CLUSTER_METHOD_NAMES
    ), "duplicate entries in _PROMPTS_CLUSTER_METHOD_NAMES"


@pytest.mark.unit
def test_prompts_controller_owns_its_cluster() -> None:
    """Every one of the 139 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    `test_ingest_controller_owns_its_cluster`.
    """
    from tldw_chatbook.UI.Library_Modules.library_prompts_controller import (
        LibraryPromptsController,
    )

    missing = [
        name
        for name in _PROMPTS_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryPromptsController, name, None))
    ]
    assert not missing, f"LibraryPromptsController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_prompt_handlers() -> None:
    """Every one of the 139 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 1
    staticmethod, to the module-level controller CLASS) -- unless a later
    cleanup task pruned it.

    Mirrors `test_screen_delegates_ingest_handlers`: a same-name forwarding
    check, not a loose "the controller is referenced somewhere" substring
    check. Skips `_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED` (empty until
    task 3) and instead asserts each such name is genuinely ABSENT from
    `LibraryScreen`, so a future accidental re-add would fail loudly here
    rather than silently reintroducing dead code.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _PROMPTS_CLUSTER_METHOD_NAMES:
        if name in _PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen but is back -- either "
                "wire it as a delegator again or drop it from "
                "_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(
            rf"_prompts_controller\.{escaped}\(", src
        ) and not re.search(rf"LibraryPromptsController\.{escaped}\(", src):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_prompts_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 1 staticmethod name in the cluster forwards to the CLASS.

    A `@staticmethod` has no `self` to reach `self._prompts_controller`
    through, so its delegator names the module-level controller class
    directly -- the conversations exemplar's own corrected shape (recipe
    §11, "the static-method delegator pattern"), reused unchanged by every
    series since.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert _PROMPTS_CLUSTER_STATICMETHOD_NAMES <= set(
        _PROMPTS_CLUSTER_METHOD_NAMES
    ), "a staticmethod name is not in the mover tuple"
    not_class_forwarding = []
    for name in _PROMPTS_CLUSTER_STATICMETHOD_NAMES:
        if name in _PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED:
            assert getattr(LibraryScreen, name, None) is None, (
                f"{name!r} was pruned from the screen but is back"
            )
            continue
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_class_forwarding.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        if not re.search(rf"LibraryPromptsController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_prompts_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_ingest_controller_exposes_every_state_field`, but through
    `prompt_state_shim_attr()` because Prompts is a THREE-prefix subsystem
    (the skills precedent) rather than Ingest's single flat prefix.
    """
    from tldw_chatbook.UI.Library_Modules.library_prompts_controller import (
        LibraryPromptsController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryPromptsState)}
    assert field_names, "state object is empty"
    missing = []
    for name in sorted(field_names):
        shim_attr = prompt_state_shim_attr(name)
        if not isinstance(
            getattr(LibraryPromptsController, shim_attr, None), property
        ):
            missing.append(shim_attr)
    assert not missing, (
        f"no prompts controller shim property found for state field(s): {missing!r}"
    )


@pytest.mark.unit
def test_prompts_controller_binds_every_name_its_moved_bodies_use() -> None:
    """Constructor-binding coverage: the byte-for-byte canon's own contract.

    A moved body is never edited, so every non-state name it spells as
    `self.<name>` (or reaches by `getattr(self, "<literal>")`) has to keep
    resolving -- on the CONTROLLER now, not the screen. This asserts the
    resolution exists at the CLASS level rather than on a constructed
    instance, deliberately: `workers` raises off the app tree (its one
    caller wraps it in `try`/`except`), so an instance probe would report a
    false failure for a binding that is in fact present and correct.

    The `focused` row is the reason this test exists at all: the skills
    series shipped a controller whose moved bodies called
    `getattr(self, "focused", None)` with no such property bound, and
    `getattr`'s default silently swallowed it into a permanent, untested
    behaviour change (recipe §3, the unbound-attribute-escape entry). Four
    Prompts movers use the identical call.
    """
    from tldw_chatbook.UI.Library_Modules.library_prompts_controller import (
        LibraryPromptsController,
    )

    assert len(_PROMPTS_CONTROLLER_BOUND_NAMES) == 42, (
        f"expected 42 bound names, got {len(_PROMPTS_CONTROLLER_BOUND_NAMES)}"
    )
    unbound = [
        name
        for name in _PROMPTS_CONTROLLER_BOUND_NAMES
        if not isinstance(getattr(LibraryPromptsController, name, None), property)
    ]
    assert not unbound, (
        "moved bodies reference these names, but the controller binds no "
        f"property for them: {unbound!r}"
    )
