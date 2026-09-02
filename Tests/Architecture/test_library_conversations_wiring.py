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


@pytest.mark.unit
def test_reader_controller_owns_its_cluster() -> None:
    from tldw_chatbook.UI.Library_Modules.library_conversation_reader_controller import (
        LibraryConversationReaderController,
    )

    for name in (
        "_load_library_conversation_reader",
        "_sync_library_conversation_reader",
    ):
        assert callable(getattr(LibraryConversationReaderController, name, None))


@pytest.mark.unit
def test_screen_delegates_reader_handlers() -> None:
    import inspect

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    src = inspect.getsource(LibraryScreen.show_library_conversation_reader_read)
    assert "_conversation_reader_controller" in src, "handler is not a delegator yet"
