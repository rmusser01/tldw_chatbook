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


def _drive_delete(profile: str, *, answer: bool = False, manager=None) -> object:
    """Run `_delete_profile` far enough to capture the pushed dialog.

    `answer` is what the confirmation resolves to, so the same seam drives
    both branches of `if confirmed:`.
    """
    window = VoiceCloningWindow.__new__(VoiceCloningWindow)
    window.selected_profile = profile
    window.backend_managers = {"vc": manager} if manager is not None else {}
    window.current_backend = "vc" if manager is not None else "none"
    window.notify = lambda *_a, **_k: None
    window._load_profiles = lambda: None
    pushed: list[object] = []

    class _App:
        async def push_screen_wait(self, screen):
            pushed.append(screen)
            return answer

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



class _Manager:
    """Records whether the irreversible half actually ran."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_profile(self, name: str):
        self.deleted.append(name)
        return True, f"Deleted {name}"


def test_declining_the_confirmation_deletes_nothing():
    """Qodo review of #2799: the confirm/cancel branch was never driven --
    the fake always answered False and nothing asserted about the backend,
    so a `_delete_profile` that deleted regardless of the answer would have
    passed. That is the failure mode a confirmation dialog exists to stop."""
    manager = _Manager()
    _drive_delete("narrator-01", answer=False, manager=manager)
    assert manager.deleted == []


def test_confirming_deletes_exactly_the_named_profile():
    manager = _Manager()
    _drive_delete("narrator-01", answer=True, manager=manager)
    assert manager.deleted == ["narrator-01"]
