"""Compatibility command palette with stable keyboard selection."""

from textual.command import CommandList, CommandPalette


class StableCommandPalette(CommandPalette):
    """Freeze an actionable result snapshot when keyboard selection begins."""

    def _action_command_list(self, action: str) -> None:
        command_list = self.query_one(CommandList)
        if (
            self._list_visible
            and command_list.option_count
            and command_list.get_option_at_index(0).id != self._NO_MATCHES
        ):
            self._cancel_gather_commands()
        super()._action_command_list(action)
