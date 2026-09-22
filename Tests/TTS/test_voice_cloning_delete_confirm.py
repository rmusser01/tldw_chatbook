"""TASK-32892 item 4: the Delete-profile confirmation could never compose.

`VoiceCloningWindow._delete_profile` defined a throwaway `ModalScreen`
inside the method whose `compose` read `self.selected_profile` -- and inside
that nested class `self` is the MODAL, which has no such attribute. The
dialog raised `AttributeError` during compose, so the Delete action showed
nothing and deleted nothing.

Gate-free by construction: no `App`, no `load_settings()` -- the window is
allocated with `__new__` and only `_delete_profile`'s own seam is driven,
which is all the defect needs.
"""

from __future__ import annotations

import asyncio

from tldw_chatbook.UI.Voice_Cloning_Window import VoiceCloningWindow


def _drive_delete(profile: str) -> object:
    """Run `_delete_profile` far enough to capture the pushed dialog."""
    window = VoiceCloningWindow.__new__(VoiceCloningWindow)
    window.selected_profile = profile
    window.backend_managers = {}
    window.current_backend = "none"
    pushed: list[object] = []

    class _App:
        async def push_screen_wait(self, screen):
            pushed.append(screen)
            return False

    original_app = VoiceCloningWindow.app
    VoiceCloningWindow.app = property(lambda _self: _App())
    try:
        asyncio.run(window._delete_profile())
    finally:
        VoiceCloningWindow.app = original_app
    assert pushed, "_delete_profile pushed no confirmation dialog"
    return pushed[0]


def test_delete_confirmation_names_the_profile_it_was_built_for():
    """The name must be bound at construction, on the OWNING window.

    Unfixed, the pushed screen carried no message at all: the name was read
    at compose time off the modal (`self.selected_profile`), which is where
    it blew up.
    """
    dialog = _drive_delete("narrator-01")

    assert "narrator-01" in getattr(dialog, "message", "")
    # And the profile name never reaches a markup parser: `[old] voice`
    # would otherwise name the wrong profile in an irreversible prompt.
    assert "narrator-01" in getattr(_drive_delete("[old] narrator-01"), "message", "")

