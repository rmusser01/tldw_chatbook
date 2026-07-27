"""Profile-store configuration and application ownership integration tests.

Task 10 extends this module with Backup All orchestration coverage. Task 9
deliberately covers only path resolution and ownership construction.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from tldw_chatbook import config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tts_profiles_db_path_defaults_to_user_data_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: default,
    )
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    assert config.get_tts_profiles_db_path() == (
        tmp_path / "tldw_chatbook_tts_profiles.db"
    )


def test_tts_profiles_custom_db_path_uses_existing_validator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    custom_path = tmp_path / "profiles" / "custom.sqlite"
    validator = Mock(wraps=config.validate_path_simple)
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            str(custom_path)
            if (section, key) == ("database", "tts_profiles_db_path")
            else default
        ),
    )
    monkeypatch.setattr(config, "validate_path_simple", validator)

    assert config.get_tts_profiles_db_path() == custom_path.resolve()
    validator.assert_called_once_with(custom_path, require_exists=False)


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../../private/profiles.sqlite",
        "/tmp/profiles.sqlite;touch-payload",
        "/tmp/profiles\x00.sqlite",
    ),
)
def test_tts_profiles_custom_db_path_rejects_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            unsafe_path
            if (section, key) == ("database", "tts_profiles_db_path")
            else default
        ),
    )

    with pytest.raises(ValueError):
        config.get_tts_profiles_db_path()


def test_tts_profiles_path_is_resolved_only_in_app_constructor() -> None:
    app_path = REPO_ROOT / "tldw_chatbook/app.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    parent_by_node = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and (
            isinstance(call.func, ast.Name)
            and call.func.id == "get_tts_profiles_db_path"
        )
    ]

    assert len(calls) == 1
    ancestor = parent_by_node[calls[0]]
    while not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
        ancestor = parent_by_node[ancestor]
    assert ancestor.name == "__init__"
