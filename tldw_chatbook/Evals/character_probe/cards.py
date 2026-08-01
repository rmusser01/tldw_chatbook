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
        character_ids: ``character_cards.id`` values, as genuine ``int``.
            Not merely documented: a handle that accepts a non-int id (e.g.
            a string that a driver happens to coerce) is exactly the kind
            of load-bearing accident this package refuses to tolerate, so
            each id's type is checked here rather than trusted from the
            caller.

    Returns:
        tuple[CardSnapshot, ...]: One snapshot per id, in the order given.
        Each snapshot's ``id`` is always the *requested* id, never merely
        echoed back from the row, so a caller can trust the snapshot is
        labelled with what it asked for.

    Raises:
        ValueError: If no ids are supplied.
        ValueError: If any id is not a genuine ``int`` (``bool`` included,
            since it is an ``int`` subclass but never a real card id) --
            mirrors ``CharacterProbeConfig``'s own id validation.
        ValueError: If a card cannot be found -- the message names the
            missing id so the caller can drop it from the bench rather than
            guessing which card vanished.
        ValueError: If the row returned for an id carries its own ``id``
            field and it disagrees with the requested id -- accepting the
            row's id here would let a misbehaving handle mislabel a
            snapshot, and a mislabelled snapshot is permanent once a run
            records it.
    """
    if not character_ids:
        raise ValueError("A character probe run needs at least one character.")
    snapshots: list[CardSnapshot] = []
    for character_id in character_ids:
        if not isinstance(character_id, int) or isinstance(character_id, bool):
            raise ValueError(
                f"character_ids must be int (character_cards.id), got "
                f"{character_id!r} of type {type(character_id).__name__}."
            )
        row = chacha_db.get_character_card_by_id(character_id)
        if not row:
            raise ValueError(
                f"Character card {character_id} could not be found; "
                "remove it from the bench or restore the card."
            )
        row_id = row.get("id", character_id)
        if row_id != character_id:
            raise ValueError(
                f"Character card handle returned id {row_id!r} for "
                f"requested id {character_id!r}; refusing to snapshot a "
                "mismatched row."
            )
        snapshots.append(
            CardSnapshot(
                id=character_id,
                name=str(row.get("name") or ""),
                **{field: str(row.get(field) or "") for field in _SNAPSHOT_FIELDS},
            )
        )
    return tuple(snapshots)
