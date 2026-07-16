"""Pure skills-command resolver for the native Console composer.

This module has no dependency on Textual, the running app, or any I/O — it
mirrors :mod:`console_command_grammar`'s purity discipline. It resolves a
bare ``/skill-name [args]`` Console draft against a caller-supplied snapshot
of skill candidates (exact case-insensitive name, then unique case-
insensitive name-prefix), and builds the
:meth:`console_command_grammar.ConsoleCommandRegistry.register_fallback_resolver`
callable a caller wires up to claim those drafts.

Callers own everything this module cannot: fetching the actual candidate
snapshot (scoped to USER-INVOCABLE + TRUSTED skills only — this module never
filters by trust or invocability itself, it only matches names), re-
resolving authoritatively at dispatch time (the fallback resolver here only
decides whether to *claim* a word so unknown words still fall through to the
existing unknown-command hint), and formatting/emitting the untrusted-skill
refusal text (:data:`SKILL_UNTRUSTED_REFUSE`) once a caller has determined a
resolved name is not currently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .console_command_grammar import KIND_FALLBACK, CommandParse

SKILL_ARGS_MAX = 4000
"""Maximum character length of skill invocation args after capping."""

SKILLS_LIST_COMMAND_NAME = "skills"
"""Canonical registered-command name for listing available skills."""

SKILL_UNTRUSTED_REFUSE = (
    'Skill "{name}" isn\'t trusted ({reason}) — review and approve it in '
    "Library ▸ Skills before running it."
)
"""Refusal copy a caller fills in (``name``, ``reason``) when a resolved
skill name is not currently trusted/invocable. Not used by any function in
this module — dispatch (which alone knows a resolved name's live trust
state) is responsible for formatting and emitting it."""

SKILLS_EMPTY_LIST_ROW = "No skills yet — create them in Library ▸ Skills."
"""Transcript system-row text `format_skills_list` returns for no candidates."""


@dataclass(frozen=True)
class SkillCommandCandidate:
    """One skill eligible for bare ``/skill-name`` resolution.

    Args:
        name: Canonical skill name (already scoped to whatever population
            the caller considers eligible, e.g. user-invocable + trusted).
        description: Short human-readable description, used only for
            `format_skills_list`'s transcript rows.
    """

    name: str
    description: str = ""


@dataclass(frozen=True)
class SkillResolution:
    """Result of resolving one ``word`` against a candidate snapshot.

    Args:
        kind: One of ``"resolved"``, ``"ambiguous"``, or ``"none"``.
        name: The matched candidate's canonical name, set only when
            ``kind == "resolved"``.
        matches: The competing candidate names, set only when
            ``kind == "ambiguous"``.
    """

    kind: str
    name: str = ""
    matches: tuple[str, ...] = ()


def resolve_skill_command(
    word: str, args: str, candidates: tuple[SkillCommandCandidate, ...]
) -> SkillResolution:
    """Resolve a bare ``/word`` Console token against a skill candidate snapshot.

    Args:
        word: The command word typed after the leading ``/`` (no slash).
        args: The remaining draft text after ``word``. Unused by the
            resolution rules themselves (callers pre-cap it via
            `cap_skill_args` before it reaches a dispatch layer) but kept in
            the signature per the interface contract.
        candidates: The skill population eligible for resolution — already
            scoped by the caller (e.g. user-invocable + trusted only).

    Returns:
        `SkillResolution` with ``kind`` set as follows: an empty or
        whitespace-only ``word`` (e.g. a bare ``/`` or ``/ `` draft, which
        `console_command_grammar` splits into an empty command word) never
        matches anything and is always `"none"` -- without this guard every
        candidate name trivially ``.startswith("")``, so an empty word
        would otherwise "match" every candidate (resolving to a lone
        candidate, or reporting `"ambiguous"` for two or more) even though
        no skill name was actually typed. Otherwise an exact case-
        insensitive name match always wins first (`"resolved"`); otherwise a
        *unique* case-insensitive name-prefix match resolves to that
        candidate's canonical name (`"resolved"`); two or more prefix
        matches are `"ambiguous"` (`matches` lists every competing name, in
        candidate order); no match at all is `"none"`.
    """
    del args  # Not used by the resolution rules; kept for interface parity.
    if not word.strip():
        return SkillResolution(kind="none")
    word_lower = word.lower()

    for candidate in candidates:
        if candidate.name.lower() == word_lower:
            return SkillResolution(kind="resolved", name=candidate.name)

    prefix_matches = tuple(
        candidate.name for candidate in candidates if candidate.name.lower().startswith(word_lower)
    )
    if len(prefix_matches) == 1:
        return SkillResolution(kind="resolved", name=prefix_matches[0])
    if len(prefix_matches) >= 2:
        return SkillResolution(kind="ambiguous", matches=prefix_matches)
    return SkillResolution(kind="none")


def cap_skill_args(args: str) -> str:
    """Trim ``args`` to `SKILL_ARGS_MAX` characters.

    Args:
        args: Raw remaining draft text following a resolved skill command.

    Returns:
        ``args`` unchanged when at or under the cap, else its first
        `SKILL_ARGS_MAX` characters.
    """
    return args[:SKILL_ARGS_MAX]


def format_skills_list(candidates: tuple[SkillCommandCandidate, ...]) -> str:
    """Render a Console transcript system-row listing available skills.

    Args:
        candidates: The skill population to list (already scoped by the
            caller), in the order they should be displayed.

    Returns:
        `SKILLS_EMPTY_LIST_ROW` when ``candidates`` is empty; otherwise one
        ``/name — description`` line per candidate (just ``/name`` when its
        description is empty), joined with newlines.
    """
    if not candidates:
        return SKILLS_EMPTY_LIST_ROW

    lines = []
    for candidate in candidates:
        if candidate.description:
            lines.append(f"/{candidate.name} — {candidate.description}")
        else:
            lines.append(f"/{candidate.name}")
    return "\n".join(lines)


def make_skill_fallback_resolver(
    candidates_getter: Callable[[], tuple[SkillCommandCandidate, ...]],
) -> Callable[[str, str], "CommandParse | None"]:
    """Build a `console_command_grammar` fallback-resolver callable.

    Args:
        candidates_getter: Zero-argument callable a caller wires to fetch a
            fresh candidate snapshot at parse time (kept as an injected
            callable rather than a plain value so this module never imports
            the real skills service — that wiring belongs to dispatch).

    Returns:
        A resolver suitable for
        `console_command_grammar.ConsoleCommandRegistry.register_fallback_resolver`.
        It claims (returns a `CommandParse`) only when `resolve_skill_command`
        finds the word plausibly matches a cached skill (`"resolved"` or
        `"ambiguous"`) — the returned `CommandParse.name` is the *typed*
        word, not the resolved candidate name, so an unmodified round-trip
        through the grammar always carries what the user actually typed;
        re-resolving authoritatively (and refusing untrusted matches) is
        dispatch's job. Any other word (no match) returns ``None`` so it
        still falls through to the existing unknown-command hint.
    """

    def resolver(word: str, rest: str) -> CommandParse | None:
        candidates = candidates_getter()
        resolution = resolve_skill_command(word, rest, candidates)
        if resolution.kind in ("resolved", "ambiguous"):
            return CommandParse(kind=KIND_FALLBACK, name=word, args=cap_skill_args(rest))
        return None

    return resolver
