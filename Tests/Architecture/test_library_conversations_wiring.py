"""Conversations extraction series: state object exists and is screen-wired."""
from __future__ import annotations

import pytest

from tldw_chatbook.UI.Library_Modules.library_conversations_state import (
    LibraryConversationsState,
)

#: Every method Task 7 moved into `LibraryConversationReaderController`, under
#: its original `LibraryScreen` name (including the five `@on`-decorated
#: handlers, which keep screen-side delegators). A module-level constant, not
#: inlined per-test, so Task 8's browse-controller wiring test can follow the
#: same shape: one full-cluster ownership test plus one full-cluster
#: delegator test, both driven off one authoritative name list.
#:
#: A prior version of this file asserted only 2 of these 21 names and one
#: delegator's source -- cheap enough to write, but it proved too narrow to
#: catch a real class of bug: task-7-report.md's "a real bug found and
#: fixed" section documents a missing import that produced a swallowed
#: `NameError` inside a method these 2 names never reached. Looping the full
#: cluster does not catch that exact bug (it was already fixed before this
#: strengthening landed), but it closes the same shape of gap for the other
#: 19 names and for any future controller in this series -- see
#: `test_reader_controller_exposes_every_state_field` below for the
#: plural-state-field variant of the same risk.
_READER_CLUSTER_METHOD_NAMES: tuple[str, ...] = (
    "_bootstrap_library_conversation_reader",
    "_conversation_reader_bootstrap_is_current",
    "_conversation_reader_list_summary",
    "_conversation_reader_record",
    "_conversation_reader_record_version",
    "_conversation_reader_request_is_current",
    "_conversation_reader_service",
    "_ensure_library_conversation_reader_selection",
    "_invalidate_library_conversation_reader_authority",
    "_load_library_conversation_reader",
    "_mirror_library_conversation_reader_preference",
    "_retry_library_conversation_reader",
    "_start_library_conversation_reader_selection",
    "_sync_library_conversation_reader",
    "_sync_library_conversation_reader_layout_from_shell",
    "library_conversation_reader_messages_synced",
    "retry_library_conversation_reader",
    "show_library_conversation_reader_info",
    "show_library_conversation_reader_read",
    "find_in_library_conversation",
    "_finish_library_conversation_find_focus",
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    import dataclasses

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryConversationsState)}
    assert field_names, "state object is empty"
    for name in field_names:
        for prefix in ("_library_conversation_", "_library_conversations_"):
            if isinstance(getattr(LibraryScreen, prefix + name, None), property):
                break
        else:
            pytest.fail(f"no screen shim property found for state field {name!r}")


@pytest.mark.unit
def test_reader_controller_owns_its_cluster() -> None:
    """Every one of the 21 moved names is a callable on the controller.

    Covers the whole cluster, not a 2-name sample: a name present on
    `LibraryScreen` (as a delegator) but missing on the controller would
    otherwise only surface as an `AttributeError` at call time, inside
    whichever delegator reached for it -- the same swallowed-failure shape
    task-7-report.md's fix-report section documents for a missing import.
    """
    from tldw_chatbook.UI.Library_Modules.library_conversation_reader_controller import (
        LibraryConversationReaderController,
    )

    missing = [
        name
        for name in _READER_CLUSTER_METHOD_NAMES
        if not callable(getattr(LibraryConversationReaderController, name, None))
    ]
    assert not missing, (
        f"LibraryConversationReaderController is missing: {missing!r}"
    )


@pytest.mark.unit
def test_screen_delegates_reader_handlers() -> None:
    """Every one of the 21 moved names is a one-line screen delegator.

    Source-substring check (not behavioral -- a delegator could still call
    the controller with the wrong arguments and this would not catch it),
    but it catches the concrete regression this test guards against: a
    cleanup pass that re-inlines a body onto the screen, or a new method
    added under one of these names that never gets wired to the controller
    at all.
    """
    import inspect

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    not_delegators = []
    for name in _READER_CLUSTER_METHOD_NAMES:
        method = getattr(LibraryScreen, name, None)
        if method is None:
            not_delegators.append(f"{name!r} (missing entirely)")
            continue
        src = inspect.getsource(method)
        if "_conversation_reader_controller" not in src:
            not_delegators.append(name)
    assert not not_delegators, (
        f"not delegators yet: {not_delegators!r}"
    )


@pytest.mark.unit
def test_reader_controller_exposes_every_state_field() -> None:
    """The controller's generated shim loop covers every state field.

    Mirrors `test_state_object_fields_match_the_shim_surface` above, but
    against `LibraryConversationReaderController` instead of `LibraryScreen`.
    Guards the concrete drift risk the review flagged: the controller module
    keeps its own `_READER_PLURAL_STATE_FIELDS` set (which field names use
    the `_library_conversations_` plural prefix vs. the singular
    `_library_conversation_` prefix), duplicated from the screen's
    `_CONVERSATIONS_PLURAL_STATE_FIELDS` (Task 6). A future task that adds a
    field to one set and not the other -- or that otherwise breaks the
    controller's shim-generation loop for one field -- would leave that
    field's name resolving under neither prefix on the controller; this
    test fails on that field by name rather than waiting for whichever
    moved body first reaches for it to raise an `AttributeError`.
    """
    import dataclasses

    from tldw_chatbook.UI.Library_Modules.library_conversation_reader_controller import (
        LibraryConversationReaderController,
    )

    field_names = {f.name for f in dataclasses.fields(LibraryConversationsState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        for prefix in ("_library_conversation_", "_library_conversations_"):
            if isinstance(
                getattr(LibraryConversationReaderController, prefix + name, None),
                property,
            ):
                break
        else:
            missing.append(name)
    assert not missing, (
        f"no controller shim property found for state field(s): {missing!r}"
    )
