from __future__ import annotations

import logging
import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook import config
from tldw_chatbook.Logging_Config import (
    PrivateRotatingFileHandler,
    _configure_private_file_logging,
)
from tldw_chatbook.Utils.private_paths import PrivatePathError


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _select_log_name(
    monkeypatch: pytest.MonkeyPatch,
    user_dir: Path,
    selected_name: str,
) -> None:
    monkeypatch.setattr(config, "get_user_data_dir", lambda: user_dir)
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            selected_name
            if section == "logging" and key == "log_filename"
            else default
        ),
    )


@pytest.mark.parametrize(
    "selected_name",
    ["", " ", ".", "..", "../outside/escape.log", "nested/escape.log", r"nested\escape.log"],
)
def test_log_filename_rejects_non_basename_values_without_creating_parents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    selected_name: str,
) -> None:
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    _select_log_name(monkeypatch, user_dir, selected_name)

    with pytest.raises(ValueError, match="basename"):
        config.get_cli_log_file_path()

    assert not (tmp_path / "outside").exists()
    assert not (user_dir / "nested").exists()


def test_log_filename_rejects_absolute_path_without_creating_its_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    outside = tmp_path / "outside"
    _select_log_name(monkeypatch, user_dir, str(outside / "escape.log"))

    with pytest.raises(ValueError, match="basename"):
        config.get_cli_log_file_path()

    assert not outside.exists()


def test_log_filename_returns_direct_child_of_secured_user_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    _select_log_name(monkeypatch, user_dir, "application.log")

    assert config.get_cli_log_file_path() == user_dir / "application.log"


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
def test_private_rotating_handler_hardens_active_and_rotated_generations(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(mode=0o755)
    active = log_dir / "application.log"
    rotated = log_dir / "application.log.1"
    active.write_text("old active\n", encoding="utf-8")
    rotated.write_text("old rotated\n", encoding="utf-8")
    active.chmod(0o644)
    rotated.chmod(0o644)

    handler = PrivateRotatingFileHandler(
        active,
        maxBytes=100,
        backupCount=2,
        encoding="utf-8",
    )
    try:
        handler.emit(logging.makeLogRecord({"msg": "new record"}))
        handler.doRollover()
    finally:
        handler.close()

    assert _mode(log_dir) == 0o700
    assert _mode(active) == 0o600
    assert _mode(rotated) == 0o600
    assert _mode(log_dir / "application.log.2") == 0o600


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
@pytest.mark.parametrize("generation", ["application.log", "application.log.1"])
def test_private_rotating_handler_rejects_symlinked_generation(
    tmp_path: Path,
    generation: str,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    outside = tmp_path / "outside-SENTINEL"
    outside.write_text("preserve", encoding="utf-8")
    (log_dir / generation).symlink_to(outside)

    with pytest.raises(PrivatePathError):
        PrivateRotatingFileHandler(
            log_dir / "application.log",
            maxBytes=100,
            backupCount=2,
            encoding="utf-8",
        )

    assert outside.read_text(encoding="utf-8") == "preserve"


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
def test_unsafe_file_sink_is_omitted_without_removing_other_handlers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    outside = tmp_path / "outside-SENTINEL"
    outside.write_text("preserve", encoding="utf-8")
    active = log_dir / "application.log"
    active.symlink_to(outside)
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_log_file_path",
        lambda: active,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Logging_Config.get_cli_setting",
        lambda section, key, default=None: default,
    )
    logger = logging.Logger("private-log-test")
    collecting = _CollectingHandler()
    logger.addHandler(collecting)

    installed = _configure_private_file_logging(logger)

    assert installed is False
    assert collecting in logger.handlers
    assert not any(
        isinstance(handler, PrivateRotatingFileHandler)
        for handler in logger.handlers
    )
    assert outside.read_text(encoding="utf-8") == "preserve"
    assert collecting.messages
    assert all("outside-SENTINEL" not in message for message in collecting.messages)
    assert max(map(len, collecting.messages)) < 200
