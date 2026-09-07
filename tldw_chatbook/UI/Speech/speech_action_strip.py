"""A Console-grammar action strip that keeps its buttons' own ids.

`CommandStrip` is the right shape for these actions and the wrong identity.
It rewrites every id it is given -- `tts-generate-btn` is mounted as
`workbench-action-tts-generate-btn` -- which is harmless where the handler
matches on the strip's own `_workbench_action_id`, and fatal here: the
Playground's handler compares `event.button.id == "tts-generate-btn"`, so a
renamed button silently never fires.

This strip is `CommandStrip`'s composition with that one behaviour removed:
same container classes, same per-button classes, same `compact`, so the
Console styling applies unchanged and only the identity differs.
"""

from __future__ import annotations

from typing import Any, Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button

from ..Workbench.workbench_state import WorkbenchAction


class SpeechActionStrip(Horizontal):
    """Text actions in a row, mounted under the ids they were given."""

    def __init__(
        self, actions: Iterable[WorkbenchAction], **kwargs: Any
    ) -> None:
        """Create the strip.

        Args:
            actions: The actions to render, in display order. Each action's
                ``id`` becomes its button's id verbatim.
            kwargs: Forwarded to ``Horizontal``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(
            classes=f"workbench-command-strip ds-toolbar {classes}".strip(),
            **kwargs,
        )
        self.actions = tuple(actions)

    def compose(self) -> ComposeResult:
        """Yield one button per action, keyed by the action's own id."""
        for action in self.actions:
            button = Button(
                action.label,
                id=action.id,
                classes=action.css_classes,
                disabled=action.disabled,
                compact=True,
            )
            button.tooltip = action.tooltip
            yield button
