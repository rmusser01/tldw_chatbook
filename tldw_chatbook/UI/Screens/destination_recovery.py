"""Shared recovery copy helpers for destination shell blocked states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationRecoveryState:
    """Taxonomy-aligned recovery state for a disabled destination action."""

    status_label: str
    unavailable_what: str
    why: str
    next_action: str
    recovery_action: str
    authority_owner: str
    stable_selector: str
    disabled_tooltip: str

    @property
    def visible_copy(self) -> str:
        return "\n".join(
            (
                self.status_label,
                f"Unavailable: {self.unavailable_what}.",
                f"Why: {self.why}.",
                f"Next: {self.next_action}",
                f"Recovery: {self.recovery_action}.",
                f"Owner: {self.authority_owner}.",
            )
        )
