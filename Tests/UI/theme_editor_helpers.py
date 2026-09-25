"""Shared test helper: reach the Settings > Theme editor (TASK-32948).

Theme opens on the picker; the editor sits behind Clone/New. A standalone
module because the Settings hub and the keyboard-journey modules import each
other's helpers, and a helper in either would close an import cycle.
"""


async def open_theme_editor(host, pilot):
    """Clone the highlighted theme from the picker and return the editor.

    Clone renames the working theme to ``<active>_copy`` and marks the editor
    modified -- callers that need a clean editor reset ``is_modified``.
    """
    host.screen.query_one("#settings-theme-list").focus()
    await pilot.press("c")
    await pilot.pause()
    await pilot.pause(0.2)
    return host.screen.query_one("#settings-theme-editor")
