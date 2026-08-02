from __future__ import annotations

import ast
from pathlib import Path

import tldw_chatbook
from tldw_chatbook.app import TldwCli


REPO_ROOT = Path(__file__).resolve().parents[2]
RETIRED_FILES = (
    "tldw_chatbook/Audio/transcription_history.py",
    "tldw_chatbook/Widgets/transcription_history_viewer.py",
    "tldw_chatbook/UI/Dictation_Window.py",
)
RETIRED_NAMES = {
    "get_user_database_path",
    "USER_DB_DIR",
    "USER_DB_PATH",
}


def test_rejected_history_modules_are_absent_and_production_app_imports() -> None:
    assert all(not (REPO_ROOT / relative).exists() for relative in RETIRED_FILES)
    assert issubclass(TldwCli, object)
    assert Path(tldw_chatbook.__file__).resolve().is_relative_to(REPO_ROOT)


def test_legacy_user_database_symbols_have_no_production_binding() -> None:
    offenders: list[tuple[str, str]] = []
    for path in sorted((REPO_ROOT / "tldw_chatbook").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in RETIRED_NAMES:
                offenders.append((path.relative_to(REPO_ROOT).as_posix(), node.id))
    assert offenders == []
