"""TASK-25821: the summary step withdrew Esc without saying so.

Steps 1-5 advertise "Esc skip setup" / "Esc exit setup" and Esc works. On the
summary the hint collapses to "Ctrl+B back" and Esc silently stops working --
the key the wizard taught for five screens becomes inert with no explanation,
and nothing tells the user how the wizard ends.

The completion action itself is NOT a defect: #setup-exit-chat already reads
"Start chatting" once provider and model are configured, and "Review provider
setup" when they are not, which is honest.
"""

from __future__ import annotations


def test_summary_hint_explains_how_setup_ends() -> None:
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SUMMARY_KEY_HINTS

    assert SUMMARY_KEY_HINTS.startswith("Ctrl+B back")
    # It must not simply drop the exit vocabulary in silence.
    assert "finish" in SUMMARY_KEY_HINTS.lower()


def test_summary_hint_does_not_promise_escape() -> None:
    """Esc does not exit from the summary, so the hint must not imply it."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SUMMARY_KEY_HINTS

    assert "esc" not in SUMMARY_KEY_HINTS.lower()
