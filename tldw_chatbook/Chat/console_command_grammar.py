"""Pure slash-command grammar for the native Console composer.

This module has no dependency on Textual, the running app, or any I/O. It
exposes a small tokenizer plus a registry that maps ``/word rest`` Console
drafts onto registered :class:`ConsoleCommand` entries, with an ordered
fallback-resolver hook reserved for future extensions (for example, a Skills
feature routing bare ``/skill-name`` drafts that do not match a built-in
command). Callers own all UI wiring, readiness gating, and paste-token
bookkeeping; this module only ever sees a plain draft-text string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

COMMAND_PREFIX = "/"
"""Leading character that marks a Console draft as a candidate slash command."""

KIND_COMMAND = "command"
KIND_FALLBACK = "fallback"
KIND_UNKNOWN = "unknown"
KIND_NOT_COMMAND = "not-command"

# Display-only markers rendered by ConsoleComposerBar._segment_display_text
# in place of a collapsed ("Pasted Text: N Characters") or armed-confirm
# ("Unfurl?") paste segment. Duplicated here as literal strings (rather than
# imported) because that widget module pulls in Textual, which this pure
# grammar module must never depend on.
_PASTE_TOKEN_MARKERS: tuple[str, ...] = ("Pasted Text: ", "Unfurl?")

PROMPT_COMMAND_NAME = "prompt"
PROMPT_COMMAND_ARGUMENT_HINT = "[name]"
PROMPT_COMMAND_HANDLER_ID = "insert-prompt"

SYSTEM_COMMAND_NAME = "system"
SYSTEM_COMMAND_ARGUMENT_HINT = "[name]"
SYSTEM_COMMAND_HANDLER_ID = "apply-system"


@dataclass(frozen=True)
class ConsoleCommand:
    """One registered Console slash command.

    Args:
        name: Canonical command word, without the leading slash (e.g. ``"prompt"``).
        argument_hint: Short human-readable argument placeholder (e.g. ``"[name]"``).
        handler_id: Stable id a caller dispatches on to run the command.
    """

    name: str
    argument_hint: str
    handler_id: str


@dataclass(frozen=True)
class CommandParse:
    """Result of parsing one Console draft against the command grammar.

    Args:
        kind: One of ``"command"``, ``"fallback"``, ``"unknown"``, or ``"not-command"``.
        name: Matched/attempted command word; empty for ``"not-command"``.
        args: Remaining text after the command word; empty when absent.
    """

    kind: str
    name: str = ""
    args: str = ""


def _contains_paste_token_marker(text: str) -> bool:
    """Return whether ``text`` embeds a collapsed/confirm paste display marker."""
    return any(marker in text for marker in _PASTE_TOKEN_MARKERS)


def _split_leading_token(text: str) -> tuple[str, str]:
    """Split ``text`` into its leading whitespace-delimited token and the rest.

    Args:
        text: Raw draft text, already known to start with ``COMMAND_PREFIX``.

    Returns:
        A ``(token, rest)`` pair. ``token`` is every character up to (but
        excluding) the first whitespace character; ``rest`` is everything
        after that single whitespace character, or ``""`` when no whitespace
        is present (the whole string is one token).
    """
    for index, character in enumerate(text):
        if character.isspace():
            return text[:index], text[index + 1 :]
    return text, ""


class ConsoleCommandRegistry:
    """Registry and parser for Console slash commands.

    Holds registered :class:`ConsoleCommand` entries plus an ordered list of
    fallback resolvers consulted (in registration order) for words that do
    not match a registered command. Fallback resolvers are the extension
    point a future Skills feature can use to route bare ``/skill-name``
    drafts without this module knowing anything about skills.
    """

    def __init__(self) -> None:
        self._commands: dict[str, ConsoleCommand] = {}
        self._fallback_resolvers: list[Callable[[str, str], CommandParse | None]] = []

    def register(self, command: ConsoleCommand) -> None:
        """Register (or replace) a command, keyed by its case-folded name."""
        self._commands[command.name.lower()] = command

    def register_fallback_resolver(
        self, resolver: Callable[[str, str], CommandParse | None]
    ) -> None:
        """Append a resolver consulted, in registration order, for unmatched words.

        Args:
            resolver: Called with ``(word, args)`` for a leading token that
                matched no registered command. Return a `CommandParse` to
                claim the draft, or ``None`` to defer to the next resolver
                (or to `"unknown"` if none claim it).
        """
        self._fallback_resolvers.append(resolver)

    def parse(self, draft_text: str) -> CommandParse:
        """Parse a Console draft against the registered command grammar.

        Args:
            draft_text: Plain composer draft text.

        Returns:
            `not-command` when ``draft_text`` does not start with
            `COMMAND_PREFIX`, or embeds a paste-token display marker.
            Otherwise the leading whitespace-delimited token (minus its
            slash) is matched case-insensitively against registered command
            names (`command`); failing that, each fallback resolver is
            offered the word and remainder (`fallback`); failing that,
            `unknown` with `name` set to the word.
        """
        if not draft_text.startswith(COMMAND_PREFIX) or _contains_paste_token_marker(
            draft_text
        ):
            return CommandParse(kind=KIND_NOT_COMMAND)

        token, rest = _split_leading_token(draft_text)
        word = token[len(COMMAND_PREFIX) :]

        command = self._commands.get(word.lower())
        if command is not None:
            return CommandParse(kind=KIND_COMMAND, name=command.name, args=rest)

        for resolver in self._fallback_resolvers:
            result = resolver(word, rest)
            if result is not None:
                return result

        return CommandParse(kind=KIND_UNKNOWN, name=word)

    def available_names(self) -> tuple[str, ...]:
        """Return registered command names, in registration order."""
        return tuple(command.name for command in self._commands.values())


def default_console_registry() -> ConsoleCommandRegistry:
    """Build the default registry with the built-in ``/prompt`` and ``/system`` commands."""
    registry = ConsoleCommandRegistry()
    registry.register(
        ConsoleCommand(
            name=PROMPT_COMMAND_NAME,
            argument_hint=PROMPT_COMMAND_ARGUMENT_HINT,
            handler_id=PROMPT_COMMAND_HANDLER_ID,
        )
    )
    registry.register(
        ConsoleCommand(
            name=SYSTEM_COMMAND_NAME,
            argument_hint=SYSTEM_COMMAND_ARGUMENT_HINT,
            handler_id=SYSTEM_COMMAND_HANDLER_ID,
        )
    )
    return registry
