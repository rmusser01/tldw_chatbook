"""Ingest extraction series: state object exists and is screen-wired.

Wave-5 Task 1 (ingest series 1/3, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; export series precedent: Tests/
Architecture/test_library_export_wiring.py, its state-PR-era shape -- the
closest match, since Ingest also needed no plural-prefix split and no
wiring-field exclusion). Every field ``LibraryIngestState`` declares must
have a matching generated property shim on ``LibraryScreen`` under the
``_library_ingest_<field>`` name -- the single prefix every Ingest field
uses (the recipe §2 ownership script found no field needing a plural
variant and no field held out as wiring, see ``library_ingest_state.py``'s
own module docstring for the full ownership analysis).
"""
from __future__ import annotations

import dataclasses

import pytest

from tldw_chatbook.UI.Library_Modules.library_ingest_state import (
    LibraryIngestState,
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryIngestState)}
    assert field_names, "state object is empty"
    assert len(field_names) == 20, f"expected 20 ingest fields, got {len(field_names)}"
    missing = []
    for name in field_names:
        shim = getattr(LibraryScreen, "_library_ingest_" + name, None)
        if not isinstance(shim, property):
            missing.append("_library_ingest_" + name)
    assert not missing, f"no screen shim property found for: {missing!r}"
