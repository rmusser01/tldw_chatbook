"""Pure collapse state for the Lab frame's two rails.

Kept separate from the widget so collapse behaviour is testable without
mounting anything, and so the frame can persist a plain value rather than
scraping widget state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The left catalog rail.
LAB_RAIL_LEFT = "rail"
#: The right inspector rail.
LAB_RAIL_INSPECTOR = "inspector"
#: Both rails, in render order.
LAB_RAILS: tuple[str, ...] = (LAB_RAIL_LEFT, LAB_RAIL_INSPECTOR)


def _validate(rail: str) -> None:
    """Reject rail names that are not one of the two real rails.

    Args:
        rail: Candidate rail name.

    Raises:
        ValueError: If ``rail`` is not in :data:`LAB_RAILS`.
    """
    if rail not in LAB_RAILS:
        raise ValueError(f"Unknown Lab rail {rail!r}; expected one of {LAB_RAILS}")


@dataclass(frozen=True)
class LabRailLayout:
    """Which of the Lab frame's rails are currently collapsed.

    Attributes:
        collapsed: Names of collapsed rails; members of :data:`LAB_RAILS`.
    """

    collapsed: frozenset[str] = field(default_factory=frozenset)

    def is_collapsed(self, rail: str) -> bool:
        """Report whether one rail is collapsed.

        Args:
            rail: Rail name, one of :data:`LAB_RAILS`.

        Returns:
            True when that rail is collapsed.

        Raises:
            ValueError: If ``rail`` is not a known rail.
        """
        _validate(rail)
        return rail in self.collapsed

    def toggle(self, rail: str) -> "LabRailLayout":
        """Return a new layout with one rail's collapse state flipped.

        Args:
            rail: Rail name, one of :data:`LAB_RAILS`.

        Returns:
            A new ``LabRailLayout``; the receiver is unchanged.

        Raises:
            ValueError: If ``rail`` is not a known rail.
        """
        _validate(rail)
        if rail in self.collapsed:
            return LabRailLayout(collapsed=frozenset(self.collapsed - {rail}))
        return LabRailLayout(collapsed=frozenset(self.collapsed | {rail}))
