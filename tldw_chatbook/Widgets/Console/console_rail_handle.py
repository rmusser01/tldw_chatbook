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

    def __init__(
        self,
        *,
        label: str,
        badge: str = "",
        button_id: str,
        badge_id: str,
        side: str,
        **kwargs: Any,
    ) -> None:
        """Create Console's rail handle.

        Forwards the base's keywords by name so a call site can only ever
        supply them once. ``open_tooltip`` is deliberately not accepted here:
        Console's tooltips are fixed strings derived from ``side``, not
        caller-supplied.

        Args:
            label: Rail name shown on the handle button. On the right side an
                inspector label is abbreviated for display; see
                ``_display_label``.
            badge: Optional secondary line under the button. On the right side
                approval and artifact copy is abbreviated to fit eleven
                columns; see ``_display_badge``.
            button_id: DOM id for the open button.
            badge_id: DOM id for the badge static.
            side: ``"left"`` for the Context rail, ``"right"`` for the
                Inspector rail. Selects the fixed tooltip and both
                abbreviation paths.
            kwargs: Forwarded to ``DestinationRailHandle``.
        """
        super().__init__(
            label=label,
            badge=badge,
            button_id=button_id,
            badge_id=badge_id,
            side=side,
            open_tooltip=(
                "Open Context rail" if side == "left" else "Open Inspector rail"
            ),
            **kwargs,
        )

    def _display_label(self) -> str:
        """Return the compact visible label; full text stays in the tooltip.

        Returns:
            ``"Inspector"`` when the right-side label is the inspector's
            canonical name, otherwise the label unchanged.
        """
        if self.side != "right":
            return self.label
        return "Inspector" if self.label == CONSOLE_RAIL_INSPECTOR_LABEL else self.label

    def _display_badge(self) -> str:
        """Return badge copy that fits the collapsed inspector affordance.

        Returns:
            On the right side, approval counts shortened to ``"<n> appr"`` and
            ``"artifact"`` to ``"art"``; any other badge, and every left-side
            badge, unchanged.
        """
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
