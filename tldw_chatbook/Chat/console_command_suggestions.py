"""Pure slash-command suggestion provider for the Console composer popup.

Mirrors the purity discipline of :mod:`console_command_grammar` and
:mod:`console_skill_resolver`: no Textual, no app state, no I/O. Callers own
all UI wiring and paste-segment gating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .console_command_grammar import (
    COMMAND_PREFIX,
    SKILLS_COMMAND_NAME,
    ConsoleCommandRegistry,
)
from .console_skill_resolver import SkillCommandCandidate

#: Which popup completion context a draft is in (TASK-24416).
CompletionContext = Literal["command", "skills_arg"]

# `\Z` (not `$`) so a trailing newline — e.g. a Shift+Enter multiline draft —
# breaks the match and leaves the completion context; the skills-arg separator
# is `[ \t]+` (not `\s+`) for the same reason, since `\s` also matches `\n`.
_COMMAND_MODE_PATTERN = re.compile(r"^/(\S*)\Z")
_SKILLS_ARG_MODE_PATTERN = re.compile(
    rf"^{COMMAND_PREFIX}{SKILLS_COMMAND_NAME}[ \t]+(\S*)\Z", re.IGNORECASE
)

# `ConsoleCommand` carries no description field, so the built-ins get their
# popup copy here; skill entries use the resolver snapshot descriptions.
_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "prompt": "Insert a saved prompt into the composer",
    "system": "Apply a saved system prompt to this session",
    "skills": "List or run a skill",
    "fewer-permission-prompts": "Reduce approval prompts for trusted tool actions",
    "prefill": "Prepare the start of the assistant's reply",
    "generate-image": "Generate an image (optionally via a chosen backend)",
    "generate-video": "Generate a video (optionally via a chosen backend)",
    "stream-video": "Stream a video from a URL into the transcript",
    "steer": "Send guidance into the RUNNING turn (plain messages queue)",
    "redirect": "Cut off the current response and re-run the turn with your correction",
    "emergency-stop": "Hold ALL new agent runs + scheduled dispatches (clear to resume)",
    "rewind": "Rewind the session to an earlier user prompt",
    "research": "Run deep research in the background; the report is delivered into this conversation",
    "help": "List the console commands, or /help <command> for one command's detail",
}

#: Shown for a registered command with no ``_COMMAND_DESCRIPTIONS`` entry --
#: the popup never renders an empty description (Console UX review 2026-08,
#: TX-05). Only non-built-in registrations (extensions, test doubles) can
#: reach this fallback: every built-in has an entry above.
COMMAND_DESCRIPTION_FALLBACK = "Custom command"

#: Same guarantee for skill rows: ``SkillCommandCandidate.description``
#: defaults to ``""`` when the resolver snapshot has no copy.
SKILL_DESCRIPTION_FALLBACK = "Run this skill"


@dataclass(frozen=True)
class CommandSuggestion:
    """One popup row.

    Args:
        insert_text: Full-draft replacement text applied on accept (note the
            trailing space, which re-triggers arg-mode for ``/skills ``).
        label: Display label, e.g. ``"/prompt"`` (command mode) or the bare
            skill name (skills-arg mode).
        description: Short human-readable description. Always non-empty:
            command-mode rows fall back to ``COMMAND_DESCRIPTION_FALLBACK``,
            skill rows to ``SKILL_DESCRIPTION_FALLBACK``.
    """

    insert_text: str
    label: str
    description: str = ""


def completion_context_for_draft(
    draft_text: str,
) -> tuple[CompletionContext, str] | None:
    """Return ``(context, prefix)`` for the draft's popup completion context.

    TASK-24416: the screen keys popup etiquette (sticky Escape dismissal,
    the bare-slash Enter guard) on WHICH completion context the draft is in
    and what filter token it carries -- not on draft text equality, which
    moves on every keystroke.

    Args:
        draft_text: Plain composer draft text.

    Returns:
        ``None`` outside any completion context; else ``("command", p)``
        for a bare command token ``/p`` or ``("skills_arg", p)`` inside
        ``/skills p``. ``p`` is the popup's filter prefix -- empty for a
        bare ``/`` (the full command list) or a bare ``/skills `` (all
        skills).
    """
    skills_arg_match = _SKILLS_ARG_MODE_PATTERN.match(draft_text)
    if skills_arg_match is not None:
        return ("skills_arg", skills_arg_match.group(1))
    command_match = _COMMAND_MODE_PATTERN.match(draft_text)
    if command_match is not None:
        return ("command", command_match.group(1))
    return None


def suggestions_for_draft(
    draft_text: str,
    registry: ConsoleCommandRegistry,
    skill_candidates: tuple[SkillCommandCandidate, ...],
    max_results: int = 200,
) -> list[CommandSuggestion] | None:
    """Compute popup suggestions for one composer draft.

    Two contexts: command mode (``^/\\S*\\Z`` — commands then skills, prefix-
    filtered) and skills-arg mode (``^/skills[ \\t]+\\S*\\Z`` — skill names
    for the first argument).

    Args:
        draft_text: Plain composer draft text.
        registry: The command registry; registered names lead the list.
        skill_candidates: Trusted, user-invocable skills eligible for
            suggestion, in display order.
        max_results: Hard cap on returned suggestions (per mode), bounding
            per-keystroke popup rebuild cost for large skill inventories.

    Returns:
        ``None`` when the draft is in no completion context (caller hides the
        popup); otherwise a possibly-empty list (empty also hides the popup).
    """
    skills_arg_match = _SKILLS_ARG_MODE_PATTERN.match(draft_text)
    if skills_arg_match is not None:
        prefix = skills_arg_match.group(1).lower()
        return [
            CommandSuggestion(
                insert_text=f"{COMMAND_PREFIX}{SKILLS_COMMAND_NAME} {candidate.name} ",
                label=candidate.name,
                description=candidate.description or SKILL_DESCRIPTION_FALLBACK,
            )
            for candidate in skill_candidates
            if candidate.name.lower().startswith(prefix)
        ][:max_results]

    command_match = _COMMAND_MODE_PATTERN.match(draft_text)
    if command_match is None:
        return None

    prefix = command_match.group(1).lower()
    command_names = registry.available_names()
    suggestions = [
        CommandSuggestion(
            insert_text=f"{COMMAND_PREFIX}{name} ",
            label=f"{COMMAND_PREFIX}{name}",
            description=_COMMAND_DESCRIPTIONS.get(
                name.lower(), COMMAND_DESCRIPTION_FALLBACK
            ),
        )
        for name in command_names
        if name.lower().startswith(prefix)
    ]
    command_names_lower = {name.lower() for name in command_names}
    suggestions.extend(
        # Bare ``/skill-name`` is NOT a registered command (the fallback
        # resolver was hard-removed in the `$`-mention migration), so skill
        # entries complete to the canonical ``/skills <name> `` invocation —
        # accepting one always yields a dispatchable draft.
        CommandSuggestion(
            insert_text=f"{COMMAND_PREFIX}{SKILLS_COMMAND_NAME} {candidate.name} ",
            label=f"{COMMAND_PREFIX}{SKILLS_COMMAND_NAME} {candidate.name}",
            description=candidate.description or SKILL_DESCRIPTION_FALLBACK,
        )
        for candidate in skill_candidates
        if candidate.name.lower().startswith(prefix)
        and candidate.name.lower() not in command_names_lower
    )
    return suggestions[:max_results]
