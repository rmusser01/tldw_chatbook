"""Keep Library runtime controllers off the registry's import-only walk."""

import os
import subprocess
import sys
from pathlib import Path


def test_preimport_defers_library_runtime_controllers(tmp_path: Path) -> None:
    """Reintroducing eager controller imports pays unused work during startup."""
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
deferred = {
    "tldw_chatbook.UI.Library_Modules.library_" + suffix + "_controller"
    for suffix in (
        "collections", "conversation_reader", "ingest", "note_import",
        "notes_sync", "prompts", "rag_search", "skills",
    )
}
assert not deferred.intersection(sys.modules), sorted(deferred.intersection(sys.modules))

# Static state restoration can precede construction on a fresh route.
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
assert LibraryScreen._restore_library_collections_page({}) == 1
assert LibraryScreen._restore_library_skills_scope({}).query == ""
assert LibraryScreen._restore_library_prompts_scope({}).query == ""
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
