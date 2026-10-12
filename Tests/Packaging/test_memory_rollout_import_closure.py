"""Keep first-use memory inspection off the affected startup paths."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("surface", ["chat", "settings"])
def test_unopened_surfaces_defer_runtime_objects(tmp_path: Path, surface: str) -> None:
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": str(root),
    }
    env.pop("TLDW_CONFIG_PATH", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
import tldw_chatbook
from tldw_chatbook.UI.Screens import chat_screen
assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path.cwd())
if sys.argv[1] == "chat":
    from tldw_chatbook.Widgets.Console import console_conversation_inspector
    assert "tldw_chatbook.Widgets.Console.console_next_send_selection" not in sys.modules
elif sys.argv[1] == "settings":
    from tldw_chatbook.UI.Screens import settings_screen
    for name in (
        "tldw_chatbook.UI.Screens.settings_web_search",
        "tldw_chatbook.UI.Screens.settings_advanced_config",
        "tldw_chatbook.Widgets.settings_web_search_panel",
        "tldw_chatbook.Widgets.settings_advanced_config_panel",
    ):
        assert name not in sys.modules, f"Unopened Settings executed category state: {name}"
""",
            surface,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
