"""Keep category-specific Settings services outside screen pre-import."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "deferred_prefix",
    [
        "tldw_chatbook.UI.Screens.settings_rag_profile_adapter",
        "tldw_chatbook.Tool_Packs",
        "tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review",
    ],
)
def test_preimport_defers_settings_category_services(
    tmp_path: Path, deferred_prefix: str
) -> None:
    """The full registry walk must not execute unopened Settings services."""
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
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
from pathlib import Path
from types import SimpleNamespace
import tldw_chatbook
from tldw_chatbook.app import TldwCli
import tldw_chatbook.UI.Screens.chat_screen

assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path.cwd())
routes = TldwCli._screen_preimport_route_order(SimpleNamespace())
assert routes
for route in routes:
    route.load_screen_class()
prefix = sys.argv[1]
loaded = sorted(name for name in sys.modules if name == prefix or name.startswith(prefix + "."))
assert not loaded, f"Pre-import executed unopened Settings services: {loaded}"
""",
            deferred_prefix,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
