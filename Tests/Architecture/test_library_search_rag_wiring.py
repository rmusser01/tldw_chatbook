"""Combined Search+RAG extraction series: state object exists and is screen-wired.

Wave-3 Task 2 (recipe: backlog/docs/library-decomposition-recipe.md;
export/collections series precedent: Tests/Architecture/
test_library_export_wiring.py / test_library_collections_wiring.py, their
state-PR-era shape; the conversations exemplar's own
test_library_conversations_wiring.py is the precedent for a state object
whose fields split across TWO shim prefixes). Every field
``LibraryRagSearchState`` declares must have a matching generated property
shim on ``LibraryScreen``, under the cluster's default ``_library_rag_``
prefix except the one name listed in ``SEARCH_PREFIXED_STATE_FIELDS``
(``history``), which uses ``_library_search_`` instead -- see that
constant's own docstring in ``library_rag_search_state.py`` for why this is
a single-source mapping rather than a second, independently-drifting copy
(the conversations exemplar's own task-8 fix-round lesson).

Unlike the conversations exemplar's own looser "either prefix works" check,
this test asserts the EXACT expected prefix per field (via
``SEARCH_PREFIXED_STATE_FIELDS``), not just that some property exists under
one of the two -- a field shimmed under the wrong prefix should fail this
test, not silently pass it.
"""
from __future__ import annotations

import dataclasses

import pytest

from tldw_chatbook.UI.Library_Modules.library_rag_search_state import (
    SEARCH_PREFIXED_STATE_FIELDS,
    LibraryRagSearchState,
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryRagSearchState)}
    assert field_names, "state object is empty"
    missing = []
    for name in field_names:
        prefix = (
            "_library_search_"
            if name in SEARCH_PREFIXED_STATE_FIELDS
            else "_library_rag_"
        )
        shim = getattr(LibraryScreen, prefix + name, None)
        if not isinstance(shim, property):
            missing.append(prefix + name)
    assert not missing, f"no screen shim property found for: {missing!r}"


@pytest.mark.unit
def test_search_prefixed_state_fields_are_real_state_fields() -> None:
    """Guards the single-source prefix mapping against drift: every name it
    lists must actually be a `LibraryRagSearchState` field (a typo or a
    stale entry here would otherwise silently shim nothing under the
    intended prefix and everything under the wrong one instead).
    """
    field_names = {f.name for f in dataclasses.fields(LibraryRagSearchState)}
    unknown = SEARCH_PREFIXED_STATE_FIELDS - field_names
    assert not unknown, f"SEARCH_PREFIXED_STATE_FIELDS names unknown fields: {unknown!r}"
