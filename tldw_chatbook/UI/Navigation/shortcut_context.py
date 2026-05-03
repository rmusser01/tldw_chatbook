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
        suffix = "" if self.available else " unavailable"
        return f"{self.key} {self.label}{suffix}"


@dataclass(frozen=True)
class ShortcutContext:
    """A replaceable set of shortcut hints for the active workflow."""

    source: str
    actions: tuple[ShortcutAction, ...]

    def render(self) -> str:
        visible = [action.render() for action in self.actions]
        return " | ".join(visible)
