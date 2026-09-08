"""Notes extraction series: state object exists and is screen-wired.

Wave-8 Task 1 (notes series 1/N, state PR; recipe: ``backlog/docs/
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
"""
from __future__ import annotations

import dataclasses

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
_EXPECTED_NOTES_STATE_FIELD_COUNT = 100

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
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryNotesState)}
    assert field_names, "state object is empty"
    assert len(field_names) == _EXPECTED_NOTES_STATE_FIELD_COUNT, (
        f"expected {_EXPECTED_NOTES_STATE_FIELD_COUNT} notes fields, "
        f"got {len(field_names)}"
    )
    missing = []
    for name in sorted(field_names):
        shim_attr = notes_state_shim_attr(name)
        if not isinstance(getattr(LibraryScreen, shim_attr, None), property):
            missing.append(shim_attr)
    assert not missing, f"no screen shim property found for: {missing!r}"


@pytest.mark.unit
def test_every_shim_reads_and_writes_its_own_state_field() -> None:
    """Each generated property is a real two-way shim, not a stub.

    A getter/setter pair that existed but bound the WRONG field (the
    closure-binding trap a `for` loop over `dataclasses.fields` invites --
    every generated property capturing the LAST field unless the name is bound
    as a default argument) would satisfy a bare `isinstance(..., property)`
    check while silently aliasing 100 names onto one field. This round-trips a
    distinct sentinel through every name to rule that out.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    screen = object.__new__(LibraryScreen)
    state = LibraryNotesState()
    screen._notes_state = state

    field_names = sorted(f.name for f in dataclasses.fields(LibraryNotesState))
    read_mismatch = []
    write_mismatch = []
    for name in field_names:
        shim_attr = notes_state_shim_attr(name)
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
