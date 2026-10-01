"""Keep first-use memory inspection and Library state off startup paths."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("surface", ["chat", "library", "settings", "state_exports"])
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
from unittest.mock import MagicMock
from pathlib import Path
import tldw_chatbook
from tldw_chatbook.UI.Screens import chat_screen
assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path.cwd())
if sys.argv[1] == "chat":
    from tldw_chatbook.Widgets.Console import console_conversation_inspector
    assert "tldw_chatbook.Widgets.Console.console_next_send_selection" not in sys.modules
elif sys.argv[1] == "state_exports":
    from tldw_chatbook.UI.Screens.library_screen import (
        LibraryIngestState, LibraryMediaState, LibraryNotesState, LibraryPromptsState,
    )
    from tldw_chatbook.UI.Library_Modules.library_ingest_state import LibraryIngestState as CanonicalIngest
    from tldw_chatbook.UI.Library_Modules.library_media_state import LibraryMediaState as CanonicalMedia
    from tldw_chatbook.UI.Library_Modules.library_notes_state import LibraryNotesState as CanonicalNotes
    from tldw_chatbook.UI.Library_Modules.library_prompts_state import LibraryPromptsState as CanonicalPrompts
    assert (LibraryIngestState, LibraryMediaState, LibraryNotesState, LibraryPromptsState) == (
        CanonicalIngest, CanonicalMedia, CanonicalNotes, CanonicalPrompts,
    )
elif sys.argv[1] == "settings":
    from tldw_chatbook.UI.Screens import settings_screen
    for name in (
        "tldw_chatbook.UI.Screens.settings_web_search",
        "tldw_chatbook.UI.Screens.settings_advanced_config",
        "tldw_chatbook.Widgets.settings_web_search_panel",
        "tldw_chatbook.Widgets.settings_advanced_config_panel",
    ):
        assert name not in sys.modules, f"Unopened Settings executed category state: {name}"
else:
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    # Conversation/export controller classmethod bindings legitimately remain module work.
    deferred = tuple("tldw_chatbook.UI.Library_Modules." + name for name in (
        "library_collections_state",
        "library_ingest_state", "library_media_state", "library_notes_state",
        "library_prompts_state", "library_rag_search_state", "library_skills_state",
        "library_media_trash_browse_controller", "library_collections_capture_controller",
        "note_session_port",
    ))
    loaded = [name for name in deferred if name in sys.modules]
    assert not loaded, f"Unopened Library executed runtime state: {loaded}"
    screen = LibraryScreen(MagicMock())
    assert all(name in sys.modules for name in deferred)
    assert screen._ingest_state is not None and screen._notes_state is not None
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
