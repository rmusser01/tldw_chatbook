"""TASK-25715: don't stack panels over a decision the app is waiting on.

With "Trace capture blocked -- Choose one action" mounted, Ctrl+K opened the
session switcher on top of it and F1 opened help on top of that: three layers
deep with the original decision still unresolved and no scrim to say which
surface owned input.

The screen already had exactly this rule for a different blocker -- every
Console hotkey action early-returns on `_console_setup_modal_blocking()`. A
mounted recovery card is the same kind of blocker and needs the same guard.
"""

from __future__ import annotations

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


def test_screen_exposes_a_decision_blocking_predicate() -> None:
    assert hasattr(ChatScreen, "_console_decision_blocking")


def test_hotkey_actions_consult_it() -> None:
    """The guard is worthless if the actions don't call it."""
    import inspect

    for name in (
        "action_open_console_session_switcher",
        "action_show_workbench_help",
    ):
        source = inspect.getsource(getattr(ChatScreen, name))
        assert "_console_decision_blocking" in source, (
            f"{name} must decline while a decision card is awaiting an answer"
        )
