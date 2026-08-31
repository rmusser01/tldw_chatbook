"""TASK-25826: the Send button must not resize while the user types.

`sync_action_state` runs on the keystroke path and sized the button from the
CURRENT label. The label gains a " | $" suffix the moment a price estimate
becomes available, so the button widened mid-typing and shifted the composer's
right edge under the cursor. Width should be stable across every label variant
the button can show for a given state.
"""

from __future__ import annotations

from tldw_chatbook.Widgets.Console.console_composer_bar import (
    send_button_width_for,
)


def test_price_suffix_does_not_change_the_width() -> None:
    assert send_button_width_for("Send") == send_button_width_for("Send | $")


def test_queue_and_send_share_a_stable_width() -> None:
    """Queue/Send swap on the same control; the edge must not jump."""
    widths = {
        send_button_width_for(label)
        for label in ("Send", "Send | $", "Queue", "Queue | $")
    }
    assert len(widths) == 1


def test_width_still_fits_the_longest_label() -> None:
    assert send_button_width_for("Send") >= len("Queue | $") + 2
