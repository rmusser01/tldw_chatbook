"""Shared recovery copy helpers for destination shell blocked states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationRecoveryState:
    """Taxonomy-aligned recovery state for a disabled destination action.

    Args:
        status_label: Short user-facing state label.
        unavailable_what: Specific workflow or control that cannot run.
        why: Immediate reason in user language.
        next_action: Concrete user step that can unblock or recover the workflow.
        recovery_action: Target route, retry action, setup action, or selection action.
        authority_owner: Owner of the capability or blocker.
        stable_selector: Stable widget selector used to expose and test this state.
        disabled_tooltip: Tooltip copy for the disabled control.
    """

    status_label: str
    unavailable_what: str
    why: str
    next_action: str
    recovery_action: str
    authority_owner: str
    stable_selector: str
    disabled_tooltip: str

    @staticmethod
    def _sentence(value: str) -> str:
        text = value.strip()
        if not text or text.endswith((".", "!", "?")):
            return text
        return f"{text}."

    @property
    def visible_copy(self) -> str:
        """Render visible multi-line recovery copy."""

        return "\n".join(
            (
                self.status_label,
                f"Unavailable: {self.unavailable_what}.",
                f"Why: {self.why}.",
                f"Next: {self._sentence(self.next_action)}",
                f"Recovery: {self.recovery_action}.",
                f"Owner: {self.authority_owner}.",
            )
        )
