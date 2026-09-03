"""TASK-25908: render the Console /help listing from the live command registry.

Presentation over data that already ships: the command names + argument hints
live on the registered :class:`ConsoleCommand`s and the one-line copy lives in
``console_command_suggestions._COMMAND_DESCRIPTIONS``. Generating the listing
from the registry means a newly registered command appears in /help without
touching this module.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

from .console_command_grammar import COMMAND_PREFIX, ConsoleCommand

# Fallback copy for a registered command with no description entry (never blank).
_HELP_DESCRIPTION_FALLBACK = "Custom command"

AvailabilityFn = Callable[[str], Optional[str]]


def _describe(name: str, descriptions: Mapping[str, str]) -> str:
    desc = descriptions.get(name)
    return desc if desc else _HELP_DESCRIPTION_FALLBACK


def _one_line(command: ConsoleCommand, descriptions: Mapping[str, str], availability_fn: Optional[AvailabilityFn]) -> str:
    hint = f" {command.argument_hint}" if command.argument_hint else ""
    desc = _describe(command.name, descriptions)
    line = f"  {COMMAND_PREFIX}{command.name}{hint} — {desc}"
    if availability_fn is not None:
        reason = availability_fn(command.name)
        if reason:
            line += f"  [unavailable: {reason}]"
    return line


def build_help_listing(
    commands: Sequence[ConsoleCommand],
    descriptions: Mapping[str, str],
    *,
    availability_fn: Optional[AvailabilityFn] = None,
    omit_unavailable: bool = False,
) -> str:
    """Render the full /help listing (AC#1/#2/#5).

    One line per registered command, taken live from ``commands``. When an
    ``availability_fn`` is given, a command it reports a reason for is either
    marked with that reason (default) or omitted (``omit_unavailable``) -- it is
    never silently listed as usable.
    """
    lines = ["Console commands:"]
    for command in commands:
        if (
            omit_unavailable
            and availability_fn is not None
            and availability_fn(command.name)
        ):
            continue
        lines.append(_one_line(command, descriptions, availability_fn))
    lines.append(f"Type {COMMAND_PREFIX}help <command> for details.")
    return "\n".join(lines)


def build_command_detail(
    commands: Sequence[ConsoleCommand],
    descriptions: Mapping[str, str],
    name: str,
    *,
    availability_fn: Optional[AvailabilityFn] = None,
) -> str:
    """Render the detail for one command, honest when unknown (AC#3)."""
    target = str(name or "").lstrip(COMMAND_PREFIX).strip().lower()
    for command in commands:
        if command.name.lower() == target:
            hint = f" {command.argument_hint}" if command.argument_hint else ""
            detail = f"{COMMAND_PREFIX}{command.name}{hint}\n{_describe(command.name, descriptions)}"
            if availability_fn is not None:
                reason = availability_fn(command.name)
                if reason:
                    detail += f"\nUnavailable right now: {reason}"
            return detail
    return (
        f"{COMMAND_PREFIX}{target or name} is not a known command. "
        f"Type {COMMAND_PREFIX}help to list commands."
    )
