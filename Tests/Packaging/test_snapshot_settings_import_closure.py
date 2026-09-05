"""Keep snapshot preferences outside the whole-registry pre-import pass."""

import os
import subprocess
import sys
from pathlib import Path


def test_screen_preimport_defers_snapshot_settings(tmp_path: Path) -> None:
    """Every route remains importable without loading snapshot preferences."""
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": str(repo_root),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from types import SimpleNamespace
from tldw_chatbook.app import TldwCli
import tldw_chatbook.UI.Screens.chat_screen

routes = TldwCli._screen_preimport_route_order(SimpleNamespace())
assert routes
for route in routes:
    route.load_screen_class()
assert "tldw_chatbook.LLM_Management.snapshot_settings" not in sys.modules
from tldw_chatbook.LLM_Management.snapshot_settings import SnapshotPreferences
assert SnapshotPreferences().keep_count == 10
""",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
