"""The Release action on a run with an unproven tool cleanup (TASK-40716).

Imported only when such a row is built, so it never loads at boot.
"""

from __future__ import annotations

import asyncio

from textual.widget import Widget
from textual.widgets import Button, Static

RELEASED_COPY = (
    "Tool slot released. The workspace stays held until you restart the app."
)


class ConsoleCleanupReleaseButton(Button):
    """Returns only the run's tool slot; the tool's workspace stays held."""

    def __init__(self, message_id: str) -> None:
        super().__init__(
            "Release tool slot",
            id=f"console-cleanup-release-{message_id}",
            classes="console-transcript-cleanup-release",
            tooltip=(
                "A tool's cleanup could not be proven. Frees this run's tool slot "
                "for other runs; the tool's workspace and storage stay held until "
                "you restart the app."
            ),
        )
        self.native_message_id = message_id

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        agent = getattr(self.screen, "_agent", None)
        if agent is None:
            return
        self.disabled = True
        # Releasing runs drain callbacks that write the agent-runs DB.
        released = await asyncio.to_thread(
            agent.release_unproven_tool_slot, self.native_message_id
        )
        if released:
            self.label = RELEASED_COPY
        else:
            self.disabled = False


def cleanup_release_widget(message_id: str, released: bool) -> Widget:
    """Build the Release action, or the note once the slot is released."""
    if released:
        return Static(
            RELEASED_COPY,
            id=f"console-cleanup-release-{message_id}",
            classes="console-transcript-cleanup-released",
            markup=False,
        )
    return ConsoleCleanupReleaseButton(message_id)
