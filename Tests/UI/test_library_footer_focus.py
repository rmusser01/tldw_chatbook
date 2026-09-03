"""The footer never advertises keys a focused text field will swallow.

Re-critique 2026-09-03 P1 (task-31223): the footer branch keyed off screen
state, not focus — with focus in the rail search it advertised
"] next in set | m | R" while those keystrokes were inserted as text (a
stray "]" corrupted the filter into a zero-match list). Keyboard-first
brand: the footer is the instrument and must not lie.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

from textual.widgets import Button, Input, TextArea

from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

_ROUTE = (
    ("/", "focus search"),
    ("F6", "next pane"),
    ("]", "next in set"),
    ("[", "prev in set"),
    ("m", "toggle reviewed"),
    ("R", "exit review"),
    ("", "2 of 6 · 1 reviewed"),
    ("esc", "focus Library"),
)


def _fake(focused) -> SimpleNamespace:
    fake = SimpleNamespace(
        _library_route_shortcuts_for_current_state=lambda: _ROUTE,
        _library_emergency_return_eligibility=lambda: SimpleNamespace(
            enabled=False
        ),
        _library_lifecycle=LibraryLifecycle.GRADUATED,
        focused=focused,
    )
    fake._library_footer_shortcuts_for_current_state = MethodType(
        LibraryScreen._library_footer_shortcuts_for_current_state, fake
    )
    return fake


def test_typing_focus_drops_swallowed_printable_keys():
    shortcuts = _fake(Input())._library_footer_shortcuts_for_current_state()
    keys = [key for key, _ in shortcuts]
    # Single printable keys are inserted as text while typing — never shown.
    assert "]" not in keys and "[" not in keys
    assert "m" not in keys and "R" not in keys and "/" not in keys
    # Keys that still work stay, and so does the informational progress chip.
    assert "esc" in keys and "F6" in keys
    assert ("", "2 of 6 · 1 reviewed") in shortcuts
    # The swap announces itself instead of silently thinning.
    assert any("typing" in label for _key, label in shortcuts)


def test_text_area_focus_gets_the_same_honesty():
    shortcuts = _fake(TextArea())._library_footer_shortcuts_for_current_state()
    keys = [key for key, _ in shortcuts]
    assert "]" not in keys and "m" not in keys


def test_non_typing_focus_keeps_the_full_set():
    shortcuts = _fake(Button())._library_footer_shortcuts_for_current_state()
    assert ("]", "next in set") in shortcuts
    assert ("m", "toggle reviewed") in shortcuts
    assert not any("typing" in label for _key, label in shortcuts)


def test_no_focus_keeps_the_full_set():
    shortcuts = _fake(None)._library_footer_shortcuts_for_current_state()
    assert ("]", "next in set") in shortcuts


def test_completed_set_footer_drops_walk_keys_and_offers_the_finish():
    """After 'All N reviewed' the footer stops advertising ] (task-31225).

    The final ] is an idempotent no-op on a complete set — advertising it
    violates the honest-footer rule (task-28005); R becomes the named next
    step, and m stays so un-marking can resume the walk.
    """
    entries = LibraryScreen._review_footer_entries("All 6 reviewed")
    keys = [key for key, _ in entries]
    assert "]" not in keys and "[" not in keys
    assert ("m", "toggle reviewed") in entries
    assert ("R", "finish review") in entries
    assert ("", "All 6 reviewed") in entries


def test_in_progress_set_footer_keeps_the_walk_keys():
    entries = LibraryScreen._review_footer_entries("2 of 6 · 1 reviewed")
    assert ("]", "next in set") in entries
    assert ("[", "prev in set") in entries
    assert ("R", "exit review") in entries
    assert ("", "2 of 6 · 1 reviewed") in entries
