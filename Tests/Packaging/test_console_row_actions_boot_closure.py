"""ADR-097: row actions load on first use, not Console construction."""

from Tests.Packaging.test_chat_persistence_import_closure import _run_isolated_python


def test_console_row_actions_are_deferred_and_cached(tmp_path):
    result = _run_isolated_python(
        tmp_path,
        """
import sys
from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

screen = ChatScreen(_build_test_app())
module = 'tldw_chatbook.UI.Console_Modules.row_actions'
assert module not in sys.modules
assert '_row_actions' not in vars(screen)
owner = screen._row_actions
assert module in sys.modules
assert type(owner) is sys.modules[module].ConsoleRowActionsController
assert screen._row_actions is owner
assert vars(screen)['_row_actions'] is owner
other = ChatScreen(_build_test_app())
assert other._row_actions is not owner
""",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
