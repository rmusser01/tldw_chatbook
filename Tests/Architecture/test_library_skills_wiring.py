"""Skills extraction series: state object exists and is screen-wired.

Wave-4 Task 1 (recipe: backlog/docs/library-decomposition-recipe.md;
export/collections/search+RAG series precedent: Tests/Architecture/
test_library_export_wiring.py / test_library_collections_wiring.py /
test_library_search_rag_wiring.py, their state-PR-era shape; the
conversations exemplar's own test_library_conversations_wiring.py is the
precedent for a state object whose fields split across multiple shim
prefixes). Every field ``LibrarySkillsState`` declares must have a matching
generated property shim on ``LibraryScreen``, resolved via
``skill_state_shim_attr()`` -- the single-source, three-way prefix mapping
(``_library_skill_`` default, ``_library_skills_`` for the plural-named
subset, bare ``_`` for the one unprefixed field, ``selected_skill_name``)
documented in ``library_skills_state.py``'s own module docstring. Unlike a
looser "either prefix works" check, this test asserts the EXACT expected
shim name per field, not just that some property exists somewhere.
"""
from __future__ import annotations

import dataclasses

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
