"""Existing private owners remain live after their screen wrappers are retired."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


@pytest.mark.parametrize("surface", ("history", "memberships"))
def test_prompt_projection_ports_resolve_replaced_owner_at_publish_time(surface):
    """A producer retained from construction must publish to the current owner."""
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    if surface == "history":
        producer = screen._library_prompt_history_controller
        port = producer._sync_view
        method_name = "_sync_library_prompt_history_region"
    else:
        producer = screen._library_prompt_collections_controller
        port = producer._sync_memberships
        method_name = "_sync_library_prompt_memberships"
    first, second = [], []
    screen._prompts_controller = SimpleNamespace(**{method_name: first.append})
    producer.invalidate()
    initial_state = (
        producer.state if surface == "history" else producer.membership_state
    )
    assert first == [initial_state]

    screen._prompts_controller = SimpleNamespace(**{method_name: second.append})
    producer.invalidate()
    current_state = (
        producer.state if surface == "history" else producer.membership_state
    )
    assert first == [initial_state]
    assert second == [current_state]
    assert (
        producer._sync_view if surface == "history" else producer._sync_memberships
    ) is port


def test_prompt_back_gate_reads_replaced_owner_and_keeps_selection_veto():
    """The live owner decides editor eligibility; selection still vetoes Back."""
    screen = LibraryScreen(SimpleNamespace(app_config={}))
    screen._prompts_controller = SimpleNamespace(
        _library_prompt_editor_active=lambda: False
    )
    assert screen.check_action("library_prompt_editor_back", ()) is False
    screen._prompts_controller = SimpleNamespace(
        _library_prompt_editor_active=lambda: True
    )
    assert screen.check_action("library_prompt_editor_back", ()) is True
    screen._prompts_state.select_mode = True
    assert screen.check_action("library_prompt_editor_back", ()) is False
