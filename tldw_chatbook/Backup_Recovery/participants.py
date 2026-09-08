"""Installed persistence maintenance participants (ADR-126)."""

from typing import Protocol

from .models import Inventory


class Participant(Protocol):
    owner_id: str

    def close_admission(self) -> None: ...

    def drain(self, deadline: float) -> bool: ...

    def resume(self) -> None: ...


def require_participant_coverage(
    inventory: Inventory, participants: tuple[Participant, ...]
) -> None:
    """Refuse uncovered paths; a capture adapter ID is not a participant.

    Passive sources currently have no exemption. Installed passive-source
    classifications require separate source evidence before relaxing this guard.
    """
    covered = {participant.owner_id for participant in participants}
    required = {item.owner for item in inventory.items if item.path is not None}
    if required - covered:
        raise ValueError("participant_missing")
