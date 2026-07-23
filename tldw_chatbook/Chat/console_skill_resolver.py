"""Pure skills-command resolver for the native Console.

This module has no dependency on Textual, the running app, or any I/O — it
mirrors :mod:`console_command_grammar`'s purity discipline. It resolves a
leading ``$skill-name [args]`` mention (or a `/skills <name>` registered-
command word) against a caller-supplied snapshot of skill candidates (exact
case-insensitive name, then unique case-insensitive name-prefix), and finds
embedded ``$skill-name`` mentions anywhere else in a draft
(:func:`find_embedded_mentions`).

Callers own everything this module cannot: fetching the actual candidate
snapshot (scoped to USER-INVOCABLE + TRUSTED skills only — this module never
filters by trust or invocability itself, it only matches names), re-
resolving authoritatively at execute time, and formatting/emitting the
untrusted-skill refusal text (:data:`SKILL_UNTRUSTED_REFUSE`) once a caller
has determined a resolved name is not currently trusted.

Hard removal (Task 4 of the `$`-mention migration): this module used to also
build a `console_command_grammar.ConsoleCommandRegistry.register_fallback_resolver`
callable claiming a bare ``/skill-name`` composer draft
(``make_skill_fallback_resolver``) — that factory has been deleted. Skill
invocation is now exclusively the `$name` mention form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SKILL_ARGS_MAX = 4000
"""Maximum character length of skill invocation args after capping."""

SKILLS_LIST_COMMAND_NAME = "skills"
"""Canonical registered-command name for listing available skills."""

MENTION_SIGIL = "$"
"""Leading character of a Codex-style skill mention (``$skill-name``)."""

_MENTION_TOKEN = re.compile(r"[A-Za-z0-9-]+")

SKILL_MENTION_SKIPPED_NOTE = (
    'Skipped "${name}" — this skill needs review before it can run. '
    "Open /skills to review it."
)

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
class SkillMention:
    """One embedded ``$skill-name`` mention found in a draft.

    Args:
        start: Index of the ``$`` sigil in the scanned text.
        end: Index one past the last token character.
        name: The matched canonical (lowercase) skill name.
    """

    start: int
    end: int
    name: str


def _code_span_mask(text: str) -> list[bool]:
    """Return a per-character mask, True inside markdown code spans.

    Fenced blocks: a line whose stripped form starts with ``````` toggles
    fence state; fence lines and everything inside are masked. Inline spans:
    on a non-fence line with an EVEN backtick count, paired backticks are
    masked inclusively (greedy left-to-right pairing — correct for well-
    formed lines). A line with an ODD backtick count is unparseable inline
    code: no pairing scheme is reliable (a stray tick shifts the pairing
    and would un-mask a genuinely guarded span), so the ENTIRE line is
    masked — failing safe, like an unclosed fence masking to end-of-text.
    """
    mask = [False] * len(text)
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            for i in range(pos, pos + len(line)):
                mask[i] = True
        elif in_fence:
            for i in range(pos, pos + len(line)):
                mask[i] = True
        elif line.count("`") % 2 == 1:
            # Odd backtick count: pairing is ambiguous — fail safe by
            # masking the whole line (over-mask, never under-mask).
            for i in range(pos, pos + len(line)):
                mask[i] = True
        else:
            i = 0
            while i < len(line):
                if line[i] == "`":
                    close = line.find("`", i + 1)
                    if close == -1:
                        # Unreachable on even-count lines (every opening
                        # tick has a closer); kept as a guard because a
                        # -1 falling through would set ``i = 0`` and loop
                        # forever if the invariant ever broke.
                        break
                    for j in range(i, close + 1):
                        mask[pos + j] = True
                    i = close + 1
                else:
                    i += 1
        pos += len(line)
    return mask


def find_embedded_mentions(
    text: str, names: frozenset[str]
) -> tuple[SkillMention, ...]:
    """Find embedded ``$skill-name`` mentions eligible for splicing.

    Exact, case-SENSITIVE matching against ``names`` (canonical lowercase
    skill names): ``$PATH`` stays literal even when a skill named ``path``
    exists. Mentions inside markdown code spans are skipped. Single pass —
    callers must never re-scan spliced output (no recursion).

    Args:
        text: The draft text to scan (the user's original message).
        names: Canonical skill names eligible for expansion.

    Returns:
        Non-overlapping mentions in document order.
    """
    mask = _code_span_mask(text)
    mentions: list[SkillMention] = []
    index = 0
    while index < len(text):
        if text[index] == MENTION_SIGIL and not mask[index]:
            match = _MENTION_TOKEN.match(text, index + 1)
            if match is not None and match.group(0) in names:
                mentions.append(
                    SkillMention(start=index, end=match.end(), name=match.group(0))
                )
                index = match.end()
                continue
        index += 1
    return tuple(mentions)


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
        candidate.name
        for candidate in candidates
        if candidate.name.lower().startswith(word_lower)
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
        ``$name — description`` line per candidate (just ``$name`` when its
        description is empty), joined with newlines.
    """
    if not candidates:
        return SKILLS_EMPTY_LIST_ROW

    lines = []
    for candidate in candidates:
        if candidate.description:
            lines.append(f"${candidate.name} — {candidate.description}")
        else:
            lines.append(f"${candidate.name}")
    return "\n".join(lines)
