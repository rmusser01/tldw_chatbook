"""Console-scoped command palette provider (posting-style).

Yields Console (ChatScreen) actions only while the Console screen is the
active screen, mirroring ``ThemeProvider`` in ``app.py`` for Hit construction.
"""

from __future__ import annotations

from functools import partial

from textual.command import Hit, Hits, Provider


class ConsoleCommandProvider(Provider):
    """Yield Console actions only while the Console screen is active."""

    def _console_screen(self):
        """Return the active ChatScreen, or ``None`` outside Console."""
        screen = self.screen
        if type(screen).__name__ != "ChatScreen":
            return None
        return screen

    def _commands(self, screen) -> tuple[tuple[str, object, str], ...]:
        """Build Console command-palette actions for ``screen``.

        Args:
            screen: Active ChatScreen exposing the Console action methods.

        Returns:
            Tuples of label, callback, and help text.
        """
        # Deferred off the boot path (app.py imports this provider eagerly):
        # both modules load with the Console screen, which is the only place
        # these commands are ever listed.
        from tldw_chatbook.UI.Console_Modules import (
            composer_run_controls as run_controls,
        )
        from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
            ACTION_ATTACH_CONTEXT,
            ACTION_IMPERSONATE,
            ACTION_IMPROVE_CURRENT_DRAFT,
            ACTION_SAVE_CHATBOOK,
        )

        return (
            (
                "Console: Recover agent work…",
                screen.action_recover_agent_work,
                "Review recorded agent work in this conversation and selected repository",
            ),
            (
                "Console: Switch session…",
                screen.action_open_console_session_switcher,
                "Fuzzy-find and activate a conversation (Ctrl+K)",
            ),
            (
                "Console: Switch model…",
                screen.action_open_console_model_popover,
                "Pick this chat's provider·model pair and quick values (Alt+M)",
            ),
            (
                "Console: New chat tab",
                screen.action_new_console_tab,
                "Open a new Console chat tab (Ctrl+T)",
            ),
            (
                "Console: New temporary chat",
                screen.action_new_temporary_console_tab,
                "Open a chat that is never saved locally",
            ),
            (
                "Console: Toggle Context rail",
                screen.action_toggle_console_context_rail,
                "Open or close the left context rail (Alt+C)",
            ),
            (
                "Console: Toggle Inspector rail",
                screen.action_toggle_console_inspector_rail,
                "Open or close the right Inspect rail (Alt+I)",
            ),
            (
                "Console: Focus composer",
                screen.action_focus_console_composer_home,
                "Return focus to the composer (Esc)",
            ),
            (
                "Console: Open Terminal",
                screen.action_open_console_terminal,
                "Open the persistent user-only host Terminal",
            ),
            (
                "Console: Switch workspace…",
                screen.action_open_console_workspace_switcher,
                "Change the active Console workspace (Alt+W)",
            ),
            (
                # TASK-2154.20 (AC-03): Alt+V had no palette path, leaving
                # default-macOS-terminal users (Option-as-Meta off) with no
                # non-Alt route to clipboard-image paste at all.
                "Console: Paste image from clipboard",
                screen.action_paste_clipboard_image,
                "Paste the clipboard image into the composer (Alt+V)",
            ),
            (
                "Console: New workspace",
                screen.action_new_console_workspace,
                "Create a local workspace and switch Console to it",
            ),
            (
                "Console: Chat settings…",
                screen.action_open_console_session_settings,
                "Tune every setting for this chat (Ctrl+O)",
            ),
            (
                "Console: Insert prompt…",
                screen.action_open_console_prompt_insert,
                "Browse saved prompts and insert one (/prompt)",
            ),
            (
                "Console: Save draft to shelf…",
                screen.action_save_console_prompt_draft,
                "Save the unsent message without sending it",
            ),
            (
                "Console: Edit system prompt",
                screen.action_open_console_system_prompt_editor,
                "Edit this session's system prompt (/system)",
            ),
            (
                "Console: Insert image style…",
                screen.action_open_console_style_insert,
                "Browse image styles and insert an @style token (/generate-image)",
            ),
            (
                "Console: View chat context",
                screen.action_view_chat_context,
                "Show current and next-send context (Ctrl+Shift+P)",
            ),
            # TASK-33625.1: keyboard routes to the viewed tab's run that do
            # not depend on the composer row's geometry.
            (
                "Console: Stop this tab's run",
                screen.action_stop_console_run,
                f"Stop the active run in this tab "
                f"({run_controls.STOP_RUN_KEY_LABEL}, /stop)",
            ),
            (
                "Console: Redirect this tab's run",
                partial(run_controls.redirect_from_draft, screen),
                "Re-run the current turn with the composer draft as the "
                "correction (/redirect)",
            ),
            # TASK-33622.2: the Composer menu and the actions that lived only
            # behind it. Each routes through the menu's own availability
            # contract (`run_composer_menu_action`).
            (
                "Console: Open composer menu",
                partial(run_controls.open_composer_menu, screen),
                "Prompts, attach, save as Chatbook, image, caption, impersonate",
            ),
            (
                "Console: Attach file…",
                partial(
                    run_controls.run_composer_menu_action, screen, ACTION_ATTACH_CONTEXT
                ),
                "Attach a file to the draft",
            ),
            (
                "Console: Save as Chatbook",
                partial(
                    run_controls.run_composer_menu_action, screen, ACTION_SAVE_CHATBOOK
                ),
                "Open the available Chatbook artifact in Artifacts",
            ),
            (
                "Console: Impersonate",
                partial(
                    run_controls.run_composer_menu_action, screen, ACTION_IMPERSONATE
                ),
                "Draft your next reply with the current model",
            ),
            (
                "Console: Improve current draft…",
                partial(
                    run_controls.run_composer_menu_action,
                    screen,
                    ACTION_IMPROVE_CURRENT_DRAFT,
                ),
                "Improve the unsent message with the current provider and model",
            ),
        )

    async def discover(self) -> Hits:
        """Yield all Console commands when the active screen is Console."""
        screen = self._console_screen()
        if screen is None:
            return
        for label, callback, help_text in self._commands(screen):
            yield Hit(1.0, label, callback, help=help_text)

    async def search(self, query: str) -> Hits:
        """Yield Console commands matching the palette query.

        Args:
            query: User-entered command-palette search text.
        """
        screen = self._console_screen()
        if screen is None:
            return
        matcher = self.matcher(query)
        for label, callback, help_text in self._commands(screen):
            score = matcher.match(label)
            if score > 0:
                yield Hit(score, matcher.highlight(label), callback, help=help_text)
