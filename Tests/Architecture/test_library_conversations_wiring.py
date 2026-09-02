"""Conversations extraction series: state object exists and is screen-wired."""
from __future__ import annotations

import pytest

from tldw_chatbook.UI.Library_Modules.library_conversations_state import (
    LibraryConversationsState,
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
