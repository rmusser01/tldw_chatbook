"""Profile-store path ownership and direct backup-manifest contract tests.

The end-to-end Backup All flow is covered with the production ``TldwCli`` in
``Tests/ProductionApp/test_tools_settings_backup.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from loguru import logger

from tldw_chatbook import config
import tldw_chatbook.UI.Tools_Settings_Window as tools_settings_module
from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow


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
        "../profiles.sqlite",
        "./../profiles.sqlite",
    ),
)
def test_tts_profiles_custom_db_path_rejects_single_parent_component_before_validation(
    monkeypatch: pytest.MonkeyPatch,
    unsafe_path: str,
) -> None:
    validator = Mock(wraps=config.validate_path_simple)
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            unsafe_path
            if (section, key) == ("database", "tts_profiles_db_path")
            else default
        ),
    )
    monkeypatch.setattr(config, "validate_path_simple", validator)

    with pytest.raises(
        ValueError,
        match="TTS profiles database path cannot contain parent traversal",
    ):
        config.get_tts_profiles_db_path()

    validator.assert_not_called()


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


def test_tts_profiles_symlink_validation_logs_no_path_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    resolved_target = tmp_path / "private-target" / "profiles.sqlite"
    resolved_target.parent.mkdir()
    resolved_target.touch()
    configured_symlink = tmp_path / "configured-profile-store.sqlite"
    configured_symlink.symlink_to(resolved_target)
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            str(configured_symlink)
            if (section, key) == ("database", "tts_profiles_db_path")
            else default
        ),
    )
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")

    try:
        result = config.get_tts_profiles_db_path()
    finally:
        logger.remove(sink_id)

    log_copy = "".join(map(str, messages))
    assert result == resolved_target.resolve()
    assert "Path resolution changed" in log_copy
    assert str(configured_symlink) not in log_copy
    assert str(resolved_target) not in log_copy


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
        and isinstance(call.func, ast.Name)
        and call.func.id == "get_tts_profiles_db_path"
    ]

    assert len(calls) == 1
    ancestor = parent_by_node[calls[0]]
    while not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
        ancestor = parent_by_node[ancestor]
    assert ancestor.name == "__init__"


def test_manifest_worker_returns_immutable_unpublished_stage(
    tmp_path: Path,
) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    publication = ToolsSettingsWindow._build_backup_manifest_publication(backup_dir)
    assert not publication.stage_path.exists()
    assert not publication.final_path.exists()

    result = ToolsSettingsWindow._write_backup_manifest(
        "20260727_010203",
        (),
        publication,
    )

    assert result == publication
    assert publication.stage_path.exists()
    assert not publication.final_path.exists()
    with pytest.raises(AttributeError):
        result.stage_path = tmp_path / "mutated"

    publication.stage_path.unlink()


def test_manifest_worker_exclusively_creates_stage_without_overwrite(
    tmp_path: Path,
) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    publication = ToolsSettingsWindow._build_backup_manifest_publication(backup_dir)
    publication.stage_path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backup_manifest_write_failed"):
        ToolsSettingsWindow._write_backup_manifest(
            "20260727_010203",
            (),
            publication,
        )

    assert publication.stage_path.read_text(encoding="utf-8") == "sentinel"
    publication.stage_path.unlink()


def test_backup_cleanup_propagates_fresh_base_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "manifest-stage.tmp"
    stage_path.touch()

    def interrupt_cleanup(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt("fresh cleanup interruption")

    monkeypatch.setattr(tools_settings_module.Path, "unlink", interrupt_cleanup)

    with pytest.raises(KeyboardInterrupt, match="fresh cleanup interruption"):
        ToolsSettingsWindow._unlink_backup_artifact(stage_path, "manifest")


def test_backup_cleanup_preserves_active_control_flow_without_exposing_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_path = tmp_path / "private-manifest-stage.tmp"
    private_error = f"cleanup failed at {stage_path}"

    def interrupt_cleanup(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt(private_error)

    monkeypatch.setattr(tools_settings_module.Path, "unlink", interrupt_cleanup)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        ToolsSettingsWindow._unlink_backup_artifact(
            stage_path,
            "manifest",
            preserve_control_flow=True,
        )
    finally:
        logger.remove(sink_id)

    public_copy = "".join(map(str, messages))
    assert "cleanup=unlink failed" in public_copy
    assert private_error not in public_copy
    assert str(stage_path) not in public_copy


def test_manifest_serialization_failure_preserves_previous_file_and_creates_no_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    manifest_path = backup_dir / "backup_info.json"
    previous_manifest = {"timestamp": "previous", "databases": []}
    manifest_path.write_text(json.dumps(previous_manifest), encoding="utf-8")
    publication = ToolsSettingsWindow._build_backup_manifest_publication(backup_dir)

    def fail_mid_dump(
        value: Any,
        stream: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        stream.write('{"timestamp": "partial')
        stream.flush()
        raise RuntimeError("private manifest serialization failure")

    monkeypatch.setattr(tools_settings_module.json, "dump", fail_mid_dump)

    with pytest.raises(RuntimeError, match="backup_manifest_write_failed"):
        ToolsSettingsWindow._write_backup_manifest(
            "20260727_010203",
            (),
            publication,
        )

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == previous_manifest
    assert not publication.stage_path.exists()
