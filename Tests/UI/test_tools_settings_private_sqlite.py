from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
import tldw_chatbook.UI.Tools_Settings_Window as settings_module
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


def _create_private_database(path: Path, *, user_version: int = 7) -> None:
    path.parent.chmod(0o700)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE settings_probe (value TEXT)")
        connection.execute("INSERT INTO settings_probe VALUES ('ok')")
        connection.execute(f"PRAGMA user_version = {user_version}")
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)


def _worker_target(database: Path) -> SimpleNamespace:
    app = SimpleNamespace(notify=MagicMock())

    def call_from_thread(callback, *args, **kwargs):
        return callback(*args, **kwargs)

    return SimpleNamespace(
        app_instance=app,
        config_data={"database": {}},
        _get_database_path=lambda _name, _config: database,
        call_from_thread=call_from_thread,
        _update_database_sizes=MagicMock(),
    )


def test_settings_vacuum_uses_registered_writable_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "settings.db"
    _create_private_database(database)
    target = _worker_target(database)
    calls: list[tuple[str, Path, bool]] = []

    def tracking_connect(owner_id, selected, *, read_only=False, **kwargs):
        calls.append((owner_id, Path(selected), read_only))
        return connect_private_sqlite(
            owner_id,
            selected,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(settings_module, "connect_private_sqlite", tracking_connect)

    settings_module.ToolsSettingsWindow._vacuum_single_worker.__wrapped__(
        target,
        "media",
    )

    assert calls == [("settings.vacuum", database, False)]
    assert any(
        "vacuumed successfully" in str(call)
        for call in target.app_instance.notify.call_args_list
    )


def test_settings_integrity_and_schema_use_registered_read_only_owners(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "settings.db"
    _create_private_database(database)
    target = _worker_target(database)
    calls: list[tuple[str, Path, bool]] = []

    def tracking_connect(owner_id, selected, *, read_only=False, **kwargs):
        calls.append((owner_id, Path(selected), read_only))
        return connect_private_sqlite(
            owner_id,
            selected,
            read_only=read_only,
            **kwargs,
        )

    monkeypatch.setattr(settings_module, "connect_private_sqlite", tracking_connect)

    settings_module.ToolsSettingsWindow._check_single_worker.__wrapped__(
        target,
        "media",
    )
    version = settings_module.ToolsSettingsWindow._get_schema_version(
        target,
        database,
    )

    assert version == 7
    assert calls == [
        ("settings.integrity", database, True),
        ("settings.schema", database, True),
    ]
    assert any(
        "integrity check passed" in str(call)
        for call in target.app_instance.notify.call_args_list
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX trust contract")
def test_settings_workers_reject_unsafe_parent_before_raw_sqlite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    database = unsafe_parent / "settings.db"
    _create_private_database(database)
    unsafe_parent.chmod(0o777)
    target = _worker_target(database)
    raw_calls: list[object] = []

    def forbidden_raw_connect(*args, **kwargs):
        raw_calls.append((args, kwargs))
        raise AssertionError("raw SQLite was reached")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", forbidden_raw_connect)
    try:
        settings_module.ToolsSettingsWindow._vacuum_single_worker.__wrapped__(
            target,
            "media",
        )
        settings_module.ToolsSettingsWindow._check_single_worker.__wrapped__(
            target,
            "media",
        )
        assert (
            settings_module.ToolsSettingsWindow._get_schema_version(target, database)
            is None
        )
    finally:
        os.chmod(unsafe_parent, 0o700)

    assert raw_calls == []
    assert (
        sum("Error" in str(call) for call in target.app_instance.notify.call_args_list)
        == 2
    )
