"""Console character/persona prompt-seed computation, loaded on use."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _character_session_prompt_seed[Seed](
    card: Mapping[str, Any],
    name_hint: str = "",
    *,
    user_name: str = "User",
    seed_type: type[Seed],
    expand_character_template: Callable[..., str],
) -> Seed:
    """Return trusted sources and safe projections from a character card.

    Joins the card's prompt-bearing fields into the Console session's system
    prompt and picks the seeded greeting from ``first_message``.

    task-1744: the join itself (field order, labels, and macro resolution)
    is ``Character_Chat_Lib.compose_character_card_text`` -- the same
    function the character-probe eval engine uses
    (``Evals.character_probe.prompt.compose_system_prompt``), so a probe run
    predicts exactly what this seeds into a real Console session. This
    includes ``message_example`` and ``post_history_instructions``, which
    Console did not send before task-1744; that is a deliberate,
    user-visible change to every character session's system prompt, not an
    incidental refactor.

    Args:
        card: The character card record.
        name_hint: Fallback display name when the card has none.

    Returns:
        The character name, exact templates, and current safe projections.
    """
    # Local import matches this module's existing convention of deferring
    # Character_Chat submodule imports (they pull in Pillow and
    # CharactersRAGDB) rather than importing them at module scope.
    from ...Character_Chat.Character_Chat_Lib import (
        compose_character_card_template,
    )

    name = str(card.get("name") or name_hint or "").strip() or "Character"
    # Cards are written against SillyTavern-style macros; compose_character_card_text
    # resolves {{char}}/{{user}} (and aliases) before the text reaches
    # session settings, or they leak verbatim into every provider payload
    # (task-1530). "User" matches the greeting-display substitution used
    # across the Personas surfaces. A card with no prompt-bearing text at
    # all falls back to a fixed instruction -- Console, not the shared
    # composer, owns that fallback, since it is not meaningful to the eval
    # engine's own empty-card handling (an intentionally blank system
    # message, see compose_system_prompt).
    system_template = (
        compose_character_card_template(
            name=name,
            system_prompt=str(card.get("system_prompt") or ""),
            personality=str(card.get("personality") or ""),
            description=str(card.get("description") or ""),
            scenario=str(card.get("scenario") or ""),
            message_example=str(card.get("message_example") or ""),
            post_history_instructions=str(card.get("post_history_instructions") or ""),
        )
        or "Stay in character."
    )
    greeting_template = str(card.get("first_message") or "")
    return seed_type(
        name=name,
        system_template=system_template,
        system_prompt=expand_character_template(
            system_template,
            user_name=user_name,
            character_name=name,
        ),
        greeting_template=greeting_template,
        greeting=expand_character_template(
            greeting_template,
            user_name=user_name,
            character_name=name,
        ),
    )


def _persona_session_prompt_seed[Seed](
    profile: Mapping[str, Any],
    name_hint: str = "",
    *,
    user_name: str = "User",
    seed_type: type[Seed],
    expand_character_template: Callable[..., str],
    sanitize_character_display_label: Callable[..., str],
) -> Seed:
    """Return the trusted persona source and its safe projection.

    Unlike character cards there is no multi-field join: a persona profile
    contributes its ``system_prompt`` field verbatim as the trusted
    template, and there is no greeting.
    """
    raw_name = str(profile.get("name") or "").strip() or str(name_hint or "").strip()
    name = sanitize_character_display_label(raw_name, max_characters=180) or "Persona"
    template = str(profile.get("system_prompt") or "")
    system_prompt = (
        expand_character_template(template, user_name=user_name, character_name=name)
        if template.strip()
        else ""
    )
    return seed_type(name=name, system_template=template, system_prompt=system_prompt)
