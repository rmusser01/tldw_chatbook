"""Ask before a quit drops staged Settings changes (TASK-33007).

``SettingsScreen.confirm_quit`` asks here before the theme editor's own
prompt, whose Discard undoes a Try: this prompt changes nothing, so its Keep
editing loses nothing. A staged change lives only in the screen's drafts, so
Ctrl+Q with "1 unsaved" used to quit at once and lose it (capture finding 22).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...Widgets.confirmation_dialog import confirm_quit_discarding_edits
from ..Screens.settings_config_models import SettingsCategoryId

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen


async def confirm_settings_quit(screen: SettingsScreen) -> bool:
    """Ask before quitting with staged changes in any Settings category.

    Theme is left out: its editor's prompt asks next. Must run inside a
    worker; the app's quit flow is one.

    Args:
        screen: The Settings screen holding the drafts.

    Returns:
        True when nothing is staged or the user chose Discard and quit; False
        to stay, including when the prompt vanished.
    """
    names = [
        summary.title
        for summary in screen._category_summaries()
        if summary.category is not SettingsCategoryId.THEME
        and screen._category_has_unsaved_changes(summary.category)
    ]
    if not names:
        return True
    return await confirm_quit_discarding_edits(
        screen,
        f"These Settings have unsaved changes: {', '.join(names)}.\n\n"
        "Keep editing to save them, or discard them and quit.",
        title="Quit and discard unsaved Settings changes?",
    )
