"""Shortcut context model for the global footer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortcutAction:
    """A single shortcut hint shown in the global footer."""

    key: str
    label: str
    available: bool = True

    def render(self) -> str:
        return f"{self.key} {self.label}"


@dataclass(frozen=True)
class ShortcutContext:
    """A replaceable set of shortcut hints for the active workflow."""

    source: str
    actions: tuple[ShortcutAction, ...]

    def render(self) -> str:
        # task-445: the footer is a single plain-text Static -- it has no way
        # to dim a hint, so the two AC-permitted options collapse to one:
        # only currently-available actions are shown at all. Previously every
        # action rendered unconditionally with a literal " unavailable" tacked
        # on, so screens with several gated actions (personas) could show a
        # footer that was MOSTLY the word "unavailable" the instant nothing
        # was selected/editing.
        visible = [action.render() for action in self.actions if action.available]
        return " | ".join(visible)
