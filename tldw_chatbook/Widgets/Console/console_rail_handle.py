"""Console's rail handle: the shared base plus Console's own vocabulary."""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Chat.console_rail_state import CONSOLE_RAIL_INSPECTOR_LABEL
from tldw_chatbook.Widgets.destination_rail import DestinationRailHandle


class ConsoleRailHandle(DestinationRailHandle):
    """Rail handle carrying Console's fixed tooltips and badge abbreviations.

    The abbreviations exist because the collapsed inspector is eleven
    columns wide. They are Console's vocabulary, not the shared base's.
    """

    def __init__(self, *, side: str, **kwargs: Any) -> None:
        super().__init__(
            side=side,
            open_tooltip=(
                "Open Context rail" if side == "left" else "Open Inspector rail"
            ),
            **kwargs,
        )

    def _display_label(self) -> str:
        """Compact visible label; full text stays in the tooltip."""
        if self.side != "right":
            return self.label
        return "Inspector" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label

    def _display_badge(self) -> str:
        """Badge copy that fits the collapsed inspector affordance."""
        if self.side != "right":
            return self.badge
        if self.badge == "1 approval":
            return "1 appr"
        if self.badge.endswith(" approvals"):
            count = self.badge.split(maxsplit=1)[0]
            return f"{count} appr"
        if self.badge == "artifact":
            return "art"
        return self.badge
