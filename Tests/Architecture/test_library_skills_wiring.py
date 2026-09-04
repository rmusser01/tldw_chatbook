"""Skills extraction series: state object and controller are screen-wired.

Wave-4 Task 1 (state PR -- skills series 1/3) and Task 2 (controller PR --
skills series 2/3; recipe: backlog/docs/library-decomposition-recipe.md;
export/collections/search+RAG series precedent: Tests/Architecture/
test_library_export_wiring.py / test_library_collections_wiring.py /
test_library_search_rag_wiring.py, their controller-PR-era shape -- no
delegator-pruning skip set yet, since that is Task 3 (cleanup)'s job for
this series; the conversations exemplar's own
test_library_conversations_wiring.py is the precedent for a state object
whose fields split across multiple shim prefixes). Every field
``LibrarySkillsState`` declares must have a matching generated property
shim on ``LibraryScreen``, resolved via ``skill_state_shim_attr()`` -- the
single-source, three-way prefix mapping (``_library_skill_`` default,
``_library_skills_`` for the plural-named subset, bare ``_`` for the one
unprefixed field, ``selected_skill_name``) documented in
``library_skills_state.py``'s own module docstring. Unlike a looser "either
prefix works" check, this test asserts the EXACT expected shim name per
field, not just that some property exists somewhere.

Task 2 adds: every one of the 86 moved names is (a) a callable on
``LibrarySkillsController`` and (b) a one-line screen delegator forwarding
to the SAME-NAMED controller method (or, for the 1 staticmethod, to the
module-level controller CLASS) -- mirroring
``test_rag_search_controller_owns_its_cluster``/
``test_screen_delegates_rag_search_handlers`` exactly, at their
controller-PR-era (pre-cleanup) shape. See
``library_skills_controller.py``'s own module docstring for the full
86-of-127 derivation and the 41 exclusions (6 merely-delegate-to-existing-
controller properties, 27 unbound-fake-self, 1 instance-attribute
monkeypatch, 1 module-globals coupling, 6 bare-self-as-identity-
argument hazard exclusions -- one found by static analysis, five found
by the verification battery after a first draft moved them and broke
real Pilot-driven / Tests/Skills tests).
"""
from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from tldw_chatbook.UI.Library_Modules.library_skills_state import (
    SKILL_UNPREFIXED_STATE_FIELDS,
    SKILLS_PLURAL_STATE_FIELDS,
    LibrarySkillsState,
    skill_state_shim_attr,
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibrarySkillsState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        shim_attr = skill_state_shim_attr(name)
        shim = getattr(LibraryScreen, shim_attr, None)
        if not isinstance(shim, property):
            missing.append(shim_attr)
    assert not missing, f"no screen shim property found for: {missing!r}"


@pytest.mark.unit
def test_skill_state_field_prefix_sets_are_real_state_fields() -> None:
    """Guards the single-source prefix mapping against drift: every name
    either exception set lists must actually be a `LibrarySkillsState`
    field (a typo or a stale entry here would otherwise silently shim
    nothing under the intended prefix and everything under the wrong one
    instead).
    """
    field_names = {f.name for f in dataclasses.fields(LibrarySkillsState)}
    unknown_plural = SKILLS_PLURAL_STATE_FIELDS - field_names
    unknown_bare = SKILL_UNPREFIXED_STATE_FIELDS - field_names
    assert not unknown_plural, (
        f"SKILLS_PLURAL_STATE_FIELDS names unknown fields: {unknown_plural!r}"
    )
    assert not unknown_bare, (
        f"SKILL_UNPREFIXED_STATE_FIELDS names unknown fields: {unknown_bare!r}"
    )


@pytest.mark.unit
def test_skill_state_prefix_sets_do_not_overlap() -> None:
    """A field can only be in at most one of the two exception sets --
    membership in both would make `skill_state_shim_attr`'s branch order
    silently pick one and hide the other's intent."""
    overlap = SKILLS_PLURAL_STATE_FIELDS & SKILL_UNPREFIXED_STATE_FIELDS
    assert not overlap, f"fields claimed by both prefix exception sets: {overlap!r}"


@pytest.mark.unit
def test_wiring_fields_stay_off_the_state_object() -> None:
    """`_library_skill_import_coordinator`/`_library_skills_browse_controller`
    are wiring (capture-controller precedent), not state -- they must NOT
    be `LibrarySkillsState` fields, and must remain plain (non-property)
    attributes on `LibraryScreen` once an instance exists.
    """
    field_names = {f.name for f in dataclasses.fields(LibrarySkillsState)}
    assert "import_coordinator" not in field_names
    assert "browse_controller" not in field_names
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert not isinstance(
        getattr(LibraryScreen, "_library_skill_import_coordinator", None), property
    )
    assert not isinstance(
        getattr(LibraryScreen, "_library_skills_browse_controller", None), property
    )


#: Every method Task 2 moved into `LibrarySkillsController`, under its
#: original `LibraryScreen` name. Derived from a full `ast` census of every
#: `LibraryScreen` method whose name contains "skill" (133 raw matches, 127
#: unique -- matching Task 1's own census), minus 6 merely-delegate-to-
#: existing-controller properties, 27 unbound-fake-self exclusions, 1
#: instance-attribute-monkeypatch exclusion, 1 module-globals-coupling
#: exclusion, and 6 bare-self-as-identity-argument hazard exclusions -- NOT a
#: prefix/substring shortcut. See `library_skills_controller.py`'s module
#: docstring for the full per-name reasoning behind every exclusion.
_SKILLS_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_apply_library_skill_detail",
    "_apply_library_skill_detail_failure",
    "_apply_library_skill_save_success",
    "_arm_library_skill_editor",
    "_begin_library_skill_trust_setup",
    "_bootstrap_library_skill_trust",
    "_build_library_skill_tool_catalog",
    "_claim_library_skill_detail_generation",
    "_consume_library_skill_scroll_pending",
    "_delete_library_skill",
    "_do_library_skill_trust_reset",
    "_enter_library_skill_conflict",
    "_enter_library_skill_create_editor",
    "_flush_library_skill_save",
    "_focus_library_skill_name",
    "_focus_library_skills_page_control",
    "_invalidate_library_skill_detail_generation",
    "_library_skill_detail_request_is_current",
    "_library_skill_on_disk_path",
    "_library_skill_text_fields_match_state",
    "_library_skill_work_pane_kwargs",
    "_library_skills_canvas_kwargs",
    "_library_skills_list_canvas_kwargs",
    "_load_library_skill_script_grant",
    "_load_library_skills_trust_posture",
    "_mark_library_skill_dirty",
    "_mirror_library_skills_reader_preference",
    "_notify_skill_dirty_veto",
    "_open_library_skill_editor_for_review",
    "_read_library_skill_editor_fields",
    "_read_library_skill_live_name",
    "_refresh_library_skill_detail",
    "_refresh_library_skill_script_grant",
    "_refresh_library_skill_trust_status",
    "_refresh_library_skills_after_committed_mutation",
    "_render_library_skill_trust_panel",
    "_request_library_skill_trust_bootstrap_passphrase",
    "_request_library_skill_trust_passphrase",
    "_restore_library_skills_scope",
    "_review_library_skill_trust",
    "_revoke_library_skill_script_grant",
    "_run_library_skill_delete",
    "_run_library_skill_save",
    "_save_library_skill",
    "_set_library_skill_discard_enabled",
    "_setup_library_skill_trust",
    "_skills_context_or_none",
    "_snapshot_library_skill_live_fields",
    "_sync_library_skill_description_hint",
    "_sync_library_skill_lifecycle_actions",
    "_sync_library_skills_browse_result",
    "_sync_library_skills_reader_layout_from_shell",
    "_unlock_library_skill_trust",
    "_update_library_skill_status_static",
    "_update_library_skill_toggle_buttons",
    "_update_library_skill_warnings_static",
    "handle_library_skill_back",
    "handle_library_skill_body_changed",
    "handle_library_skill_cancel",
    "handle_library_skill_conflict_reload",
    "handle_library_skill_context_toggle",
    "handle_library_skill_description_changed",
    "handle_library_skill_detail_retry",
    "handle_library_skill_disable_model_toggle",
    "handle_library_skill_discard",
    "handle_library_skill_editor_mode",
    "handle_library_skill_input_changed",
    "handle_library_skill_more_actions",
    "handle_library_skill_name_changed",
    "handle_library_skill_reader_mode",
    "handle_library_skill_save",
    "handle_library_skill_script_grant_revoke",
    "handle_library_skill_tool_filter",
    "handle_library_skill_tool_selection",
    "handle_library_skill_trust_approve",
    "handle_library_skill_trust_setup",
    "handle_library_skill_trust_unlock",
    "handle_library_skill_trust_view_details",
    "handle_library_skill_user_invocable_toggle",
    "handle_library_skills_import_path_submitted",
    "handle_library_skills_import_run",
    "handle_library_skills_page_next",
    "handle_library_skills_page_previous",
    "handle_library_skills_retry",
    "handle_library_skills_trust_reset_cancel",
    "handle_library_skills_trust_reset_confirm",
)

#: The 1 name above that is a `@staticmethod` on `LibraryScreen`. Its
#: delegator forwards straight to the module-level `LibrarySkillsController`
#: CLASS (per the conversations/export/collections/search+RAG wiring
#: tests' "static-method delegator pattern" precedent), not through
#: `self._skills_controller`.
_SKILLS_CLUSTER_STATICMETHOD_NAMES: frozenset[str] = frozenset(
    {
        "_restore_library_skills_scope",
    }
)


@pytest.mark.unit
def test_skills_cluster_method_names_are_genuinely_skill_named() -> None:
    """Guards the hand-kept cluster list against drift with the census: every
    name must contain "skill" (case-insensitive) -- a typo here would
    silently test the wrong surface.
    """
    not_skill_named = [n for n in _SKILLS_CLUSTER_METHOD_NAMES if "skill" not in n.lower()]
    assert not not_skill_named, f"non-skill-named cluster entries: {not_skill_named!r}"
    assert len(_SKILLS_CLUSTER_METHOD_NAMES) == 86, (
        f"expected 86 moved names, got {len(_SKILLS_CLUSTER_METHOD_NAMES)}"
    )


@pytest.mark.unit
def test_skills_controller_owns_its_cluster() -> None:
    """Every one of the 86 moved names is a callable on the controller.

    Covers the whole cluster, not a hand-picked sample -- mirrors
    `test_rag_search_controller_owns_its_cluster`.
    """
    from tldw_chatbook.UI.Library_Modules.library_skills_controller import (
        LibrarySkillsController,
    )

    missing = [
        name
        for name in _SKILLS_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibrarySkillsController, name, None))
    ]
    assert not missing, f"LibrarySkillsController is missing: {missing!r}"


@pytest.mark.unit
def test_screen_delegates_skills_handlers() -> None:
    """Every one of the 86 moved names is a one-line screen delegator that
    forwards to the SAME-NAMED controller method (or, for the 1
    staticmethod, to the module-level controller CLASS).

    Mirrors `test_screen_delegates_rag_search_handlers`: a same-name
    forwarding check, not a loose "the controller is referenced somewhere"
    substring check. No delegators are pruned yet at this (controller-PR)
    stage -- that is Task 3 (cleanup)'s job.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _SKILLS_CLUSTER_METHOD_NAMES:
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        escaped = re.escape(name)
        if not re.search(
            rf"_skills_controller\.{escaped}\(", src
        ) and not re.search(rf"LibrarySkillsController\.{escaped}\(", src):
            not_delegators.append(name)
    assert not not_delegators, f"not delegators yet: {not_delegators!r}"


@pytest.mark.unit
def test_skills_cluster_staticmethods_forward_to_the_controller_class() -> None:
    """The 1 staticmethod name in the cluster forwards to the CLASS, not an instance."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_class_forwarding = []
    for name in _SKILLS_CLUSTER_STATICMETHOD_NAMES:
        src = inspect.getsource(getattr(LibraryScreen, name))
        if not re.search(rf"LibrarySkillsController\.{re.escape(name)}\(", src):
            not_class_forwarding.append(name)
    assert not not_class_forwarding, (
        f"expected class-forwarding delegators: {not_class_forwarding!r}"
    )


@pytest.mark.unit
def test_skills_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_rag_search_controller_exposes_every_state_field`, using
    the three-way prefix mapping `skill_state_shim_attr` resolves.
    """
    from tldw_chatbook.UI.Library_Modules.library_skills_controller import (
        LibrarySkillsController,
    )

    field_names = {f.name for f in dataclasses.fields(LibrarySkillsState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        shim_attr = skill_state_shim_attr(name)
        if not isinstance(getattr(LibrarySkillsController, shim_attr, None), property):
            missing.append(shim_attr)
    assert not missing, (
        f"no skills controller shim property found for state field(s): {missing!r}"
    )
