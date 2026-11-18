"""The Release action on an unproven tool cleanup (TASK-40716, TASK-41307).

Imported only when such a row is built, so it never loads at boot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Button, Static

RELEASED_COPY = (
    "Tool slot released. The workspace stays held until you restart the app."
)


class ConsoleCleanupReleaseButton(Button):
    """Returns only the tool slot; the tool's workspace stays held."""

    def __init__(
        self, message_id: str, release: Callable[[str], bool] | None = None
    ) -> None:
        super().__init__(
            "Release tool slot",
            id=f"console-cleanup-release-{message_id}",
            classes="console-transcript-cleanup-release",
            tooltip=(
                "A tool's cleanup could not be proven. Frees its tool slot "
                "for other work; the tool's workspace and storage stay held "
                "until you restart the app."
            ),
        )
        self.native_message_id = message_id
        self._release = release

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        release = self._release or getattr(
            getattr(self.screen, "_agent", None), "release_unproven_tool_slot", None
        )
        if release is None:
            return
        self.disabled = True
        # Releasing runs drain callbacks that write the agent-runs DB.
        released = await asyncio.to_thread(release, self.native_message_id)
        if not released:
            self.disabled = False
            return
        self.label = RELEASED_COPY
        # The 0.2 s poll stops when idle; repaint the held-slots segment now.
        resync = getattr(
            getattr(self.screen, "_agent", None), "resync_held_tool_slots", None
        )
        if callable(resync):
            resync()


def cleanup_release_widget(
    message_id: str,
    released: bool,
    explanation: str = "",
    release: Callable[[str], bool] | None = None,
) -> Widget:
    """Build the Release action, or the note once the slot is released.

    ``explanation`` names the unproven cleanup above the action, for a turn
    that stopped before its run existed and so has no explanation row.
    """
    if released:
        action: Widget = Static(
            RELEASED_COPY,
            id=f"console-cleanup-release-{message_id}",
            classes="console-transcript-cleanup-released",
            markup=False,
        )
    else:
        action = ConsoleCleanupReleaseButton(message_id, release)
    if not explanation:
        return action
    note = Static(
        explanation, classes="console-transcript-cleanup-released", markup=False
    )
    box = Vertical(note, action, classes="console-transcript-cleanup-released")
    box.styles.margin = 0
    return box
