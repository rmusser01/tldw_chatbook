"""Collections extraction series: the state object is screen-wired.

Wave-2 Task 5 (state PR -- collections series 1/3; recipe:
backlog/docs/library-decomposition-recipe.md; export series precedent:
Tests/Architecture/test_library_export_wiring.py, its own state-PR-era
shape). Every field LibraryCollectionsState declares must have a matching
generated property shim on LibraryScreen under the
`_library_collections_<field>` name -- the single prefix every Collections
field uses (unlike Conversations, no field needed a different prefix
variant; the ownership analysis found none, see the task-5 report).

Scope matches the export series' own Task 2 precedent exactly: this is the
state PR only ("state-object fields <-> shim surface"), not the full-cluster
controller-ownership/same-name-delegator-forwarding shape a later
controller PR in this series will add.
"""
from __future__ import annotations

import dataclasses

import pytest

from tldw_chatbook.UI.Library_Modules.library_collections_state import (
    LibraryCollectionsState,
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryCollectionsState)}
    assert field_names, "state object is empty"
    for name in field_names:
        shim = getattr(LibraryScreen, "_library_collections_" + name, None)
        assert isinstance(shim, property), (
            f"no screen shim property found for state field {name!r}"
        )
