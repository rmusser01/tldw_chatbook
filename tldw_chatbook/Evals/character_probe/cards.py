"""Read-only card access across the ChaChaNotes/Evals database boundary."""

from __future__ import annotations

from typing import Any, Sequence

from .models import CardSnapshot

#: Card fields copied into a run. Every one participates in prompt assembly,
#: and adding a field here is only HALF the change --
#: ``prompt.compose_system_prompt`` must compose it too, or the snapshot
#: grows text the model never sees.
#:
#: What is excluded is the card's non-prompting columns (``image``, the
#: ``created_at``/``last_modified`` timestamps, ``version``/``client_id``/
#: ``deleted``) plus the metadata columns no probe path reads today
#: (``creator_notes``, ``alternate_greetings``, ``tags``, ``creator``,
#: ``character_version``, ``extensions``). Omitting a field from this list
#: is NOT automatically harmless: ``description`` -- the primary V2 persona
#: field, which Console itself sends -- was missing here until the
#: whole-branch review of task-1691 phase 1, and every probe until then ran
#: against a character stripped of its main definition. Anything omitted in
#: future must be omitted because it is not prompting text, not because a
#: list happened not to mention it.
_SNAPSHOT_FIELDS = (
    "description",
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
