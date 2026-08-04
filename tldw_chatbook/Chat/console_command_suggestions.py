"""Pure slash-command suggestion provider for the Console composer popup.

Mirrors the purity discipline of :mod:`console_command_grammar` and
:mod:`console_skill_resolver`: no Textual, no app state, no I/O. Callers own
all UI wiring and paste-segment gating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .console_command_grammar import (
    COMMAND_PREFIX,
    SKILLS_COMMAND_NAME,
    ConsoleCommandRegistry,
)
from .console_skill_resolver import SkillCommandCandidate

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
    "prefill": "Arm, pin, or clear a response prefill",
    "generate-image": "Generate an image (optionally via a chosen backend)",
    "rewind": "Rewind the session to an earlier user prompt",
}


@dataclass(frozen=True)
class CommandSuggestion:
    """One popup row.

    Args:
        insert_text: Full-draft replacement text applied on accept (note the
            trailing space, which re-triggers arg-mode for ``/skills ``).
        label: Display label, e.g. ``"/prompt"`` (command mode) or the bare
            skill name (skills-arg mode).
        description: Short human-readable description; may be empty.
    """

    insert_text: str
    label: str
    description: str = ""


def suggestions_for_draft(
    draft_text: str,
    registry: ConsoleCommandRegistry,
    skill_candidates: tuple[SkillCommandCandidate, ...],
) -> list[CommandSuggestion] | None:
    """Compute popup suggestions for one composer draft.

    Returns ``None`` when the draft is in no completion context (caller hides
    the popup); otherwise a possibly-empty list (empty also hides the popup).
    Two contexts: command mode (``^/\\S*\\Z`` — commands then skills, prefix-
    filtered) and skills-arg mode (``^/skills[ \\t]+\\S*\\Z`` — skill names
    for the first argument).
    """
    skills_arg_match = _SKILLS_ARG_MODE_PATTERN.match(draft_text)
    if skills_arg_match is not None:
        prefix = skills_arg_match.group(1).lower()
        return [
            CommandSuggestion(
                insert_text=f"{COMMAND_PREFIX}{SKILLS_COMMAND_NAME} {candidate.name} ",
                label=candidate.name,
                description=candidate.description,
            )
            for candidate in skill_candidates
            if candidate.name.lower().startswith(prefix)
        ]

    command_match = _COMMAND_MODE_PATTERN.match(draft_text)
    if command_match is None:
        return None

    prefix = command_match.group(1).lower()
    command_names = registry.available_names()
    suggestions = [
        CommandSuggestion(
            insert_text=f"{COMMAND_PREFIX}{name} ",
            label=f"{COMMAND_PREFIX}{name}",
            description=_COMMAND_DESCRIPTIONS.get(name.lower(), ""),
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
            description=candidate.description,
        )
        for candidate in skill_candidates
        if candidate.name.lower().startswith(prefix)
        and candidate.name.lower() not in command_names_lower
    )
    return suggestions
