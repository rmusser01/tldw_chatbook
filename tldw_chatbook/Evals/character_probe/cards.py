"""Read-only card access across the ChaChaNotes/Evals database boundary."""

from __future__ import annotations

from typing import Any, Sequence

from .models import CardSnapshot

#: Card fields copied into a run. Every one participates in prompt assembly;
#: anything not listed here (images, timestamps, versions) is deliberately
#: excluded because it cannot change what the model sees.
_SNAPSHOT_FIELDS = (
    "system_prompt",
    "personality",
    "scenario",
    "first_message",
    "post_history_instructions",
    "message_example",
)


def snapshot_cards(chacha_db: Any, character_ids: Sequence[int]) -> tuple[CardSnapshot, ...]:
    """Copy each requested card's prompting text, in the requested order.

    Args:
        chacha_db: A ``CharactersRAGDB``-shaped handle; only
            ``get_character_card_by_id`` is used, so a fake needs just that.
        character_ids: ``character_cards.id`` values, as INTEGERs.

    Returns:
        tuple[CardSnapshot, ...]: One snapshot per id, in the order given.

    Raises:
        ValueError: If no ids are supplied, or a card cannot be found -- the
            message names the missing id so the caller can drop it from the
            bench rather than guessing which card vanished.
    """
    if not character_ids:
        raise ValueError("A character probe run needs at least one character.")
    snapshots: list[CardSnapshot] = []
    for character_id in character_ids:
        row = chacha_db.get_character_card_by_id(character_id)
        if not row:
            raise ValueError(
                f"Character card {character_id} could not be found; "
                "remove it from the bench or restore the card."
            )
        snapshots.append(
            CardSnapshot(
                id=int(row.get("id", character_id)),
                name=str(row.get("name") or ""),
                **{field: str(row.get(field) or "") for field in _SNAPSHOT_FIELDS},
            )
        )
    return tuple(snapshots)
