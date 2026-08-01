"""Assembling the messages a character probe sends.

Steering composes AHEAD of the card's own system prompt: steering is a
model-level instruction ("answer in English") and the card is the content it
operates on. Both are preserved -- silently dropping either would evaluate
something other than what the bench describes.

Card text is written against SillyTavern-style macros, so ``{{char}}``,
``{{user}}`` and their aliases are RESOLVED here before the text leaves this
module. Console resolves them for exactly the same reason (task-1530: they
otherwise leak verbatim into every provider payload), and a probe that
shipped a literal ``{{user}}`` to the model would be evaluating text no real
chat with that card ever produces.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .models import CardSnapshot

#: What ``{{user}}`` resolves to. A probe run has no human in the exchange --
#: the "user" turns are a script -- so there is no real name to use, and
#: "User" is what Console substitutes on the Personas surfaces too, which
#: keeps a probe's prompt comparable with a real session's.
USER_MACRO_NAME = "User"

#: What ``{{char}}`` resolves to when a card has no usable name. Matches
#: ``Character_Chat_Lib.replace_placeholders``'s own fallback.
FALLBACK_CHAR_NAME = "Character"


def resolve_card_macros(text: str, card: CardSnapshot) -> str:
    """Resolve ``{{char}}``/``{{user}}`` (and aliases) in one card's text.

    Args:
        text: Text taken from the card.
        card: The card the text came from; its ``name`` supplies
            ``{{char}}``.

    Returns:
        str: The text with every macro substituted. ``""`` in, ``""`` out.
    """
    # Local import, matching chat_screen.py's own convention for this
    # function: Character_Chat_Lib imports Pillow and CharactersRAGDB at
    # module scope, and this engine package stays importable without paying
    # for either until a prompt is actually composed.
    from ...Character_Chat.Character_Chat_Lib import replace_placeholders

    return replace_placeholders(
        text, card.name.strip() or FALLBACK_CHAR_NAME, USER_MACRO_NAME
    )


def compose_system_prompt(card: CardSnapshot, steering: Optional[str]) -> str:
    """Build the system prompt for one card under one target's steering.

    Every field on :class:`~tldw_chatbook.Evals.character_probe.models.CardSnapshot`
    is composed here, because ``cards.py`` snapshots a field precisely when
    it "participates in prompt assembly". That includes ``description`` (the
    primary V2 persona field) and ``message_example`` -- example dialogue
    shapes a character's voice as much as its personality does, and leaving
    either out would silently narrow what the probe evaluates.

    The card->prompt join itself (field order, labels, and macro
    resolution) lives in
    ``Character_Chat_Lib.compose_character_card_text`` -- the ONE joiner
    shared with Console's own session seeding
    (``UI.Screens.chat_screen._character_session_prompt_seed``, task-1744).
    This function's own job is narrower: attach ``steering``, which is not
    card text and has no Console equivalent.

    Macros are resolved in the CARD's text only. ``steering`` is the eval
    author's own model-level instruction rather than card text, so it is
    passed through verbatim -- a ``{{char}}`` written there is not a card
    macro and must not silently acquire one card's name in a run that spans
    several.

    Args:
        card: The snapshotted card.
        steering: The target's own system prompt, or None when unsteered.

    Returns:
        str: Steering first, then the card's persona text with macros
        resolved. Empty parts are omitted rather than contributing blank
        lines. If every part is empty (no steering, no card text at all),
        this deliberately returns ``""`` rather than raising:
        ``build_messages`` always emits exactly one leading system message,
        so the message shape stays stable even for a content-free card.
    """
    # Local import, matching this module's existing convention (see
    # resolve_card_macros): Character_Chat_Lib imports Pillow and
    # CharactersRAGDB at module scope, and this engine package stays
    # importable without paying for either until a prompt is actually
    # composed.
    from ...Character_Chat.Character_Chat_Lib import compose_character_card_text

    card_text = compose_character_card_text(
        name=card.name.strip() or FALLBACK_CHAR_NAME,
        system_prompt=card.system_prompt,
        personality=card.personality,
        description=card.description,
        scenario=card.scenario,
        message_example=card.message_example,
        post_history_instructions=card.post_history_instructions,
        user_name=USER_MACRO_NAME,
    )
    parts = [steering or "", card_text]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def build_messages(
    card: CardSnapshot,
    steering: Optional[str],
    scripted_turns: Sequence[str],
    replies_so_far: Sequence[str],
) -> list[dict[str, str]]:
    """Build the message list for the next turn of a conversation.

    The card's ``first_message`` seeds an opening assistant turn as it does in
    real roleplay, with its macros resolved like the rest of the card's text.
    A card without one starts at the user's first scripted turn -- no greeting
    is invented, because inventing one would evaluate text the character never
    had.

    Scripted user turns are sent VERBATIM, macros included: a probe's turns
    are the eval author's text, not the card's, and the probe format's own
    rule is that turn text is reproduced exactly because prompt formatting
    changes model behaviour.

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
        messages.append(
            {
                "role": "assistant",
                "content": resolve_card_macros(card.first_message, card),
            }
        )
    for index, reply in enumerate(replies_so_far):
        messages.append({"role": "user", "content": scripted_turns[index]})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": scripted_turns[len(replies_so_far)]})
    return messages
