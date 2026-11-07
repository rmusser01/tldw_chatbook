"""The Console rename dialog loads only when its public export is used."""

from pathlib import Path

from Tests.Packaging.test_app_import_diet_closure import _run_isolated_python


def test_console_rename_modal_loads_on_first_public_export(tmp_path: Path) -> None:
    result = _run_isolated_python(
        tmp_path,
        """
import sys
import tldw_chatbook.Widgets.Console as widgets
import tldw_chatbook.UI.Console_Modules.session

module_name = "tldw_chatbook.Widgets.Console.console_rename_session_modal"
assert module_name not in sys.modules, "Rename dialog loaded before first use"
assert "ConsoleRenameSessionModal" in widgets.__all__
from tldw_chatbook.Widgets.Console import ConsoleRenameSessionModal
from tldw_chatbook.Widgets.Console.console_rename_session_modal import (
    ConsoleRenameSessionModal as implementation,
)
assert ConsoleRenameSessionModal is implementation
assert widgets.ConsoleRenameSessionModal is implementation
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
