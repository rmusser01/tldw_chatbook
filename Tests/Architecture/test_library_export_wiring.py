"""Export extraction series: state object exists and is screen-wired.

Wave-2 Task 2 (recipe: backlog/docs/library-decomposition-recipe.md;
conversations series precedent: Tests/Architecture/
test_library_conversations_wiring.py, its state-PR-era shape). Every field
LibraryExportState declares must have a matching generated property shim on
LibraryScreen under the `_library_export_<field>` name -- the single prefix
every export field uses (unlike conversations, no field needed a plural
`_library_exports_` variant; the ownership analysis found none, see the
export series' task-2 report).
"""
from __future__ import annotations

import dataclasses

import pytest

from tldw_chatbook.UI.Library_Modules.library_export_state import (
    LibraryExportState,
)


@pytest.mark.unit
def test_state_object_fields_match_the_shim_surface() -> None:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    field_names = {f.name for f in dataclasses.fields(LibraryExportState)}
    assert field_names, "state object is empty"
    for name in field_names:
        shim = getattr(LibraryScreen, "_library_export_" + name, None)
        assert isinstance(shim, property), (
            f"no screen shim property found for state field {name!r}"
        )
