"""Assembling the messages a character probe sends.

Steering composes AHEAD of the card's own system prompt: steering is a
model-level instruction ("answer in English") and the card is the content it
operates on. Both are preserved -- silently dropping either would evaluate
something other than what the bench describes.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .models import CardSnapshot


def compose_system_prompt(card: CardSnapshot, steering: Optional[str]) -> str:
    """Build the system prompt for one card under one target's steering.

    ``message_example`` is included deliberately, not merely because it is
    present on ``CardSnapshot``: ``cards.py`` snapshots it precisely because
    "every one participates in prompt assembly", and example dialogue shapes
    a character's voice as much as its personality or scenario does. Leaving
    it out would silently narrow what the probe actually evaluates.

    Args:
        card: The snapshotted card.
        steering: The target's own system prompt, or None when unsteered.

    Returns:
        str: Steering first, then the card's persona text. Empty parts are
        omitted rather than contributing blank lines.
    """
    parts = [
        steering or "",
        card.system_prompt,
        f"Personality: {card.personality}" if card.personality else "",
        f"Scenario: {card.scenario}" if card.scenario else "",
        f"Example dialogue:\n{card.message_example}" if card.message_example else "",
        card.post_history_instructions,
    ]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def build_messages(
    card: CardSnapshot,
    steering: Optional[str],
    scripted_turns: Sequence[str],
    replies_so_far: Sequence[str],
) -> list[dict[str, str]]:
    """Build the message list for the next turn of a conversation.

    The card's ``first_message`` seeds an opening assistant turn as it does in
    real roleplay. A card without one starts at the user's first scripted turn
    -- no greeting is invented, because inventing one would evaluate text the
    character never had.

    Args:
        card: The snapshotted card.
        steering: The target's own system prompt, or None.
        scripted_turns: All of the probe's user turns.
        replies_so_far: The model's replies to the preceding turns; its length
            determines which scripted turn comes next.

    Returns:
        list[dict[str, str]]: ``role``/``content`` messages, ending with the
        user turn awaiting a reply.

    Raises:
        ValueError: If ``replies_so_far`` already has as many entries as
            ``scripted_turns`` (or more). There is no next scripted turn to
            append in that case -- the conversation is already complete, and
            indexing past the end would otherwise raise an ``IndexError``
            deep inside assembly, far from the caller's actual mistake.
    """
    if len(replies_so_far) >= len(scripted_turns):
        raise ValueError(
            f"replies_so_far has {len(replies_so_far)} entries but the probe "
            f"has only {len(scripted_turns)} scripted turn(s); there is no "
            "next turn to build a message for -- the conversation is already "
            "complete."
        )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": compose_system_prompt(card, steering)}
    ]
    if card.first_message:
        messages.append({"role": "assistant", "content": card.first_message})
    for index, reply in enumerate(replies_so_far):
        messages.append({"role": "user", "content": scripted_turns[index]})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": scripted_turns[len(replies_so_far)]})
    return messages
