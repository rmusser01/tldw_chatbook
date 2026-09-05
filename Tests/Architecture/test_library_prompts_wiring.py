"""Prompts extraction series: state object exists and is screen-wired.

Wave-6 Task 1 (prompts series 1/3, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; skills series precedent: ``Tests/
Architecture/test_library_skills_wiring.py``, its state-PR-era shape -- the
closest match, since Prompts is likewise a THREE-prefix subsystem). Every
field ``LibraryPromptsState`` declares must have a matching generated
property shim on ``LibraryScreen``, resolved via ``prompt_state_shim_attr()``
-- the single-source three-way prefix mapping (``_library_prompt_`` default,
``_library_prompts_`` for the plural-named subset, bare ``_`` for the one
unprefixed field, ``selected_prompt_id``) documented in
``library_prompts_state.py``'s own module docstring. Unlike a looser "either
prefix works" check, these tests assert the EXACT expected shim name per
field, and that each shim genuinely reads AND writes the state object rather
than merely existing as a property.

The state module under test is ``tldw_chatbook.UI.Library_Modules.library_
prompts_state`` -- NOT the unrelated, pre-existing ``tldw_chatbook.Library.
library_prompts_state`` domain module of the same basename (see that
docstring's own note on the collision).
"""
from __future__ import annotations

import dataclasses

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
