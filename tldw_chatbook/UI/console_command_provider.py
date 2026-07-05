"""Console-scoped command palette provider (posting-style).

Yields Console (ChatScreen) actions only while the Console screen is the
active screen, mirroring ``ThemeProvider`` in ``app.py`` for Hit construction.
"""

from __future__ import annotations

from textual.command import Hit, Hits, Provider


class ConsoleCommandProvider(Provider):
    """Yield Console actions only while the Console screen is active."""

    def _console_screen(self):
        screen = self.screen
        if type(screen).__name__ != "ChatScreen":
            return None
        return screen

    def _commands(self, screen) -> tuple[tuple[str, object, str], ...]:
        return (
            ("Console: Switch session…", screen.action_open_console_session_switcher,
             "Fuzzy-find and activate a conversation (Ctrl+K)"),
            ("Console: Change model…", screen.action_open_console_model_popover,
             "Quick provider/model/temperature switch (Alt+M)"),
            ("Console: New chat tab", screen.action_new_console_tab,
             "Open a new Console chat tab (Ctrl+T)"),
            ("Console: Focus composer", screen.action_focus_console_composer_home,
             "Return focus to the composer (Esc)"),
            ("Console: Session settings…",
             lambda: screen.run_worker(screen._open_console_settings(), exclusive=False),
             "Open the full session settings modal"),
        )

    async def discover(self) -> Hits:
        screen = self._console_screen()
        if screen is None:
            return
        for label, callback, help_text in self._commands(screen):
            yield Hit(1.0, label, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        screen = self._console_screen()
        if screen is None:
            return
        matcher = self.matcher(query)
        for label, callback, help_text in self._commands(screen):
            score = matcher.match(label)
            if score > 0:
                yield Hit(score, matcher.highlight(label), callback, help=help_text)
