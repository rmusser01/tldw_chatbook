"""Shared test helper: reach the Settings > Theme editor (TASK-32948).

Theme opens on the picker; the editor sits behind Clone/New. A standalone
module because the Settings hub and the keyboard-journey modules import each
other's helpers, and a helper in either would close an import cycle.
"""

from __future__ import annotations

from typing import Any

from textual.pilot import Pilot


async def open_theme_editor(host: Any, pilot: Pilot) -> Any:
    """Clone the highlighted theme from the picker and return the editor.

    Clone renames the working theme to ``<active>_copy`` and leaves the editor
    clean (TASK-32948 PR 2) -- callers that need it dirty make a real edit.

    Args:
        host: The running test app whose current screen is Settings ▸ Theme.
        pilot: The pilot driving ``host``.

    Returns:
        The mounted ``SettingsThemeEditor`` (``#settings-theme-editor``).
    """
    host.screen.query_one("#settings-theme-list").focus()
    await pilot.press("c")
    await pilot.pause()
    await pilot.pause(0.2)
    return host.screen.query_one("#settings-theme-editor")
