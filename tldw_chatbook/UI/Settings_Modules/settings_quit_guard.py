"""Ask before a quit drops staged Settings changes (TASK-33007).

``SettingsScreen.confirm_quit`` asks here before the theme editor's own
prompt, whose Discard undoes a Try: this prompt changes nothing, so its Keep
editing loses nothing. A staged change lives only in the screen's drafts, so
Ctrl+Q with "1 unsaved" used to quit at once and lose it (capture finding 22).

Leaving Settings keeps its drafts in ``app.screen_state_store`` until the
next visit (ruling C-N1). While Settings is not open, ``KeptSettingsDraft``
stands in for it in the quit walk (``quit_confirmation_screens``) and asks
the same question; Keep editing goes back to Settings, where the draft is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...Constants import TAB_SETTINGS
from ...Widgets.confirmation_dialog import confirm_quit_discarding_edits
from ...Widgets.settings_theme_editor import ThemeLeaveModal
from ..Navigation.main_navigation import NavigateToScreen
from ..Screens.settings_config_models import SettingsCategoryId

if TYPE_CHECKING:
    from ..Screens.settings_screen import SettingsScreen

#: The snapshot key ``SettingsScreen.save_state`` names its unsaved
#: categories under, for ``kept_settings_draft``.
KEPT_UNSAVED_TITLES = "unsaved_titles"


def unsaved_settings_titles(screen: SettingsScreen) -> list[str]:
    """Name every category with unsaved changes, Theme aside.

    Theme is left out: its editor's prompt asks on its own, and leaving
    Settings never keeps a theme edit.

    Args:
        screen: The Settings screen holding the drafts.

    Returns:
        The categories' titles, in rail order.
    """
    return [
        summary.title
        for summary in screen._category_summaries()
        if summary.category is not SettingsCategoryId.THEME
        and screen._category_has_unsaved_changes(summary.category)
    ]


async def _ask_to_discard(owner: Any, names: list[str]) -> bool:
    return await confirm_quit_discarding_edits(
        owner,
        f"These Settings have unsaved changes: {', '.join(names)}.\n\n"
        "Keep editing to save them, or discard them and quit.",
        title="Quit and discard unsaved Settings changes?",
    )


async def confirm_settings_quit(screen: SettingsScreen) -> bool:
    """Ask before quitting with staged changes in any Settings category.

    Theme is left out: its editor's prompt asks next. Must run inside a
    worker; the app's quit flow is one.

    Args:
        screen: The Settings screen holding the drafts.

    Returns:
        True when nothing is staged or the user chose Discard and quit; False
        to stay, including when the prompt vanished or a theme leave prompt
        is already open.
    """
    if screen._theme_leave_in_progress or isinstance(
        screen.app.screen, ThemeLeaveModal
    ):
        return False  # one prompt at a time, as confirm_navigation
    names = unsaved_settings_titles(screen)
    if not names:
        return True
    return await _ask_to_discard(screen, names)


class KeptSettingsDraft:
    """Settings' place in the quit walk while its draft is only kept."""

    def __init__(self, app: Any, names: list[str]) -> None:
        self.app = app
        self._names = names

    async def confirm_quit(self) -> bool:
        """Ask as Settings would; Keep editing goes back to Settings.

        Returns:
            True when the user chose Discard and quit; False to stay.
        """
        if await _ask_to_discard(self, self._names):
            return True
        self.app.post_message(NavigateToScreen(TAB_SETTINGS))
        return False


def kept_settings_draft(app: Any, screens: list[Any]) -> KeptSettingsDraft | None:
    """Return a stand-in for Settings when a kept draft has unsaved changes.

    Args:
        app: The running app.
        screens: The screens the quit walk already asks.

    Returns:
        None while Settings is among ``screens`` (its own ``confirm_quit``
        asks about the live drafts; the store may still hold the snapshot
        it opened from), when the app keeps no Settings snapshot for the
        current runtime, or when that snapshot has nothing unsaved.
    """
    if any(getattr(screen, "screen_name", None) == TAB_SETTINGS for screen in screens):
        return None
    store = getattr(app, "screen_state_store", None)
    identity = getattr(app, "_current_runtime_identity", None)
    if store is None or not callable(identity):
        return None
    snapshot = store.restore(TAB_SETTINGS, identity()) or {}
    names = snapshot.get(KEPT_UNSAVED_TITLES)
    if not isinstance(names, list) or not names:
        return None
    return KeptSettingsDraft(app, names)
