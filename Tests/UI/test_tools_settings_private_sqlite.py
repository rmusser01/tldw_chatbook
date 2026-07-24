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
from tldw_chatbook.DB.private_sqlite import (
    copy_private_sqlite,
    restore_private_sqlite,
)


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
        app=SimpleNamespace(call_from_thread=call_from_thread),
        app_instance=app,
        config_data={"database": {}},
        _get_database_path=lambda _name, _config: database,
        call_from_thread=call_from_thread,
        _update_database_sizes=MagicMock(),
        _update_last_backup_status=MagicMock(),
        _get_schema_version=lambda _path: 7,
    )


@pytest.mark.asyncio
async def test_settings_raw_toml_save_delegates_to_locked_config_replacement(
    monkeypatch,
) -> None:
    parsed = {"general": {"users_name": "private-user"}}
    loaded = {"general": {"users_name": "private-user"}, "defaults": {}}
    replace = MagicMock(return_value=loaded)
    monkeypatch.setattr(settings_module, "replace_cli_config", replace)
    target = SimpleNamespace(
        query_one=lambda *_args: SimpleNamespace(
            text='[general]\nusers_name = "private-user"\n'
        ),
        app_instance=SimpleNamespace(notify=MagicMock()),
        config_data={},
    )

    await settings_module.ToolsSettingsWindow._save_raw_toml_config(target)

    replace.assert_called_once_with(parsed)
    assert target.config_data == loaded


@pytest.mark.asyncio
async def test_settings_config_export_delegates_to_private_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "config_backup.toml"
    export = MagicMock(return_value=snapshot)
    monkeypatch.setattr(settings_module, "export_cli_config_snapshot", export)
    target = SimpleNamespace(
        app_instance=SimpleNamespace(notify=MagicMock()),
        config_data={"API": {"openai_api_key": "secret"}},
    )

    await settings_module.ToolsSettingsWindow._export_configuration(target)

    export.assert_called_once_with(target.config_data)
    assert str(snapshot) in str(target.app_instance.notify.call_args)


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


@pytest.mark.skipif(os.name != "posix", reason="POSIX backup privacy contract")
def test_settings_single_backup_secures_directory_database_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    database = tmp_path / "custom" / "media.db"
    database.parent.mkdir(mode=0o755)
    _create_private_database(database)
    database.parent.chmod(0o755)
    target = _worker_target(database)
    calls: list[tuple[str, Path, Path]] = []

    def tracking_copy(owner_id, source, destination):
        calls.append((owner_id, Path(source), Path(destination)))
        return copy_private_sqlite(owner_id, source, destination)

    monkeypatch.setattr(
        settings_module,
        "copy_private_sqlite",
        tracking_copy,
        raising=False,
    )
    previous = os.umask(0)
    try:
        settings_module.ToolsSettingsWindow._backup_single_worker.__wrapped__(
            target,
            "media",
        )
    finally:
        os.umask(previous)

    backup_dir = home / ".local" / "share" / "tldw_cli" / "backups" / "media"
    backups = list(backup_dir.glob("media_backup_*.db"))
    metadata = list(backup_dir.glob("media_backup_*.json"))
    assert len(backups) == len(metadata) == 1
    assert calls == [("settings.single_backup", database, backups[0])]
    assert (backup_dir.stat().st_mode & 0o777) == 0o700
    assert (backups[0].stat().st_mode & 0o777) == 0o600
    assert (metadata[0].stat().st_mode & 0o777) == 0o600
    assert database.parent.stat().st_mode & 0o777 == 0o755


@pytest.mark.skipif(os.name != "posix", reason="POSIX backup privacy contract")
def test_settings_bulk_backup_uses_checked_copies_and_private_info(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    sources = {
        "chachanotes": tmp_path / "chachanotes.db",
        "prompts": tmp_path / "prompts.db",
        "media": tmp_path / "media.db",
        "evals": tmp_path / "evals.db",
        "rag": tmp_path / "rag.db",
        "subscriptions": tmp_path / "subscriptions.db",
    }
    for source in sources.values():
        _create_private_database(source)
    target = _worker_target(sources["media"])
    target.config_data = {"database": {}}
    target._get_database_path = lambda name, _config: sources[name]
    calls: list[tuple[str, Path, Path]] = []

    def tracking_copy(owner_id, source, destination):
        calls.append((owner_id, Path(source), Path(destination)))
        return copy_private_sqlite(owner_id, source, destination)

    monkeypatch.setattr(
        settings_module,
        "copy_private_sqlite",
        tracking_copy,
        raising=False,
    )
    previous = os.umask(0)
    try:
        settings_module.ToolsSettingsWindow._backup_worker.__wrapped__(target)
    finally:
        os.umask(previous)

    backup_root = home / ".local" / "share" / "tldw_cli" / "backups"
    backup_dirs = list(backup_root.iterdir())
    assert len(backup_dirs) == 1
    backup_dir = backup_dirs[0]
    assert len(calls) == 6
    assert {call[0] for call in calls} == {"settings.bulk_backup"}
    assert {call[1] for call in calls} == set(sources.values())
    assert (backup_dir.stat().st_mode & 0o777) == 0o700
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in backup_dir.iterdir())


def test_settings_bulk_maintenance_covers_every_database_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sources = {
        name: tmp_path / f"{name}.db"
        for name in (
            "chachanotes",
            "prompts",
            "media",
            "evals",
            "rag",
            "subscriptions",
        )
    }
    for source in sources.values():
        _create_private_database(source)
    target = _worker_target(sources["media"])
    target._get_database_path = lambda name, _config: sources[name]
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

    settings_module.ToolsSettingsWindow._vacuum_worker.__wrapped__(target)
    settings_module.ToolsSettingsWindow._integrity_worker.__wrapped__(target)

    assert calls == [
        *[("settings.vacuum", source, False) for source in sources.values()],
        *[("settings.integrity", source, True) for source in sources.values()],
    ]


def test_settings_size_refresh_covers_every_database_owner(tmp_path: Path) -> None:
    sources = {
        name: tmp_path / f"{name}.db"
        for name in (
            "chachanotes",
            "prompts",
            "media",
            "evals",
            "rag",
            "subscriptions",
        )
    }
    for source in sources.values():
        source.write_bytes(b"x")
    widgets = {f"#db-size-{name}": MagicMock() for name in sources}
    target = SimpleNamespace(
        config_data={"database": {}},
        _get_database_path=lambda name, _config: sources[name],
        _format_file_size=lambda size: f"{size} B",
        query_one=lambda selector: widgets[selector],
    )

    settings_module.ToolsSettingsWindow._update_database_sizes(target)

    for widget in widgets.values():
        widget.update.assert_called_once_with("Size: 1 B")


def test_settings_database_path_router_uses_canonical_runtime_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    user_data_dir = tmp_path / "runtime-data"
    expected = {
        "chachanotes": tmp_path / "chachanotes.db",
        "prompts": tmp_path / "prompts.db",
        "media": tmp_path / "media.db",
        "evals": user_data_dir / "evals.db",
        "rag": user_data_dir / "rag_indexing.db",
        "subscriptions": tmp_path / "subscriptions.db",
    }
    for name in ("chachanotes", "prompts", "media", "subscriptions"):
        monkeypatch.setattr(
            settings_module,
            f"get_{name}_db_path",
            lambda name=name: expected[name],
            raising=False,
        )
    monkeypatch.setattr(
        settings_module,
        "get_user_data_dir",
        lambda: user_data_dir,
        raising=False,
    )

    target = SimpleNamespace()
    selected = {
        name: settings_module.ToolsSettingsWindow._get_database_path(target, name, {})
        for name in expected
    }

    assert selected == expected
    assert (
        settings_module.ToolsSettingsWindow._get_database_path(target, "unknown", {})
        is None
    )


def test_settings_chatbook_import_paths_match_importer_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = {
        "ChaChaNotes": str(tmp_path / "chachanotes.db"),
        "Prompts": str(tmp_path / "prompts.db"),
        "Media": str(tmp_path / "media.db"),
    }
    monkeypatch.setattr(
        settings_module,
        "get_chatbook_database_paths",
        lambda: expected,
    )
    target = SimpleNamespace()

    selected = settings_module.ToolsSettingsWindow._get_chatbook_import_database_paths(
        target,
        {},
    )

    assert selected == expected


def test_settings_chatbook_worker_uses_explicit_import_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_chatbook.Chatbooks.chatbook_importer import ImportStatus

    database = tmp_path / "settings.db"
    target = _worker_target(database)
    db_paths = {
        "ChaChaNotes": str(tmp_path / "chachanotes.db"),
        "Prompts": str(tmp_path / "prompts.db"),
        "Media": str(tmp_path / "media.db"),
    }
    captured: dict[str, object] = {}

    class FakeImporter:
        def __init__(self, selected_paths):
            captured["db_paths"] = selected_paths

        def import_chatbook(self, **kwargs):
            status = kwargs["import_status"]
            status.successful_items = 3
            status.skipped_items = 1
            status.failed_items = 0
            status.add_warning("test warning")
            captured["status"] = status
            return True, "Successfully imported 3 items"

    monkeypatch.setattr(
        "tldw_chatbook.Chatbooks.chatbook_importer.ChatbookImporter",
        FakeImporter,
    )

    settings_module.ToolsSettingsWindow._import_chatbook_worker.__wrapped__(
        target,
        str(tmp_path / "bundle.zip"),
        db_paths,
    )

    assert captured["db_paths"] == db_paths
    assert isinstance(captured["status"], ImportStatus)
    assert any(
        "Successfully imported 3 items (1 skipped, 0 failed)" in str(call)
        for call in target.app_instance.notify.call_args_list
    )
    assert not any(
        "Error during import" in str(call)
        for call in target.app_instance.notify.call_args_list
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX export privacy contract")
@pytest.mark.parametrize(
    ("worker_name", "result_method", "records", "file_pattern"),
    [
        (
            "_export_conversations_worker",
            "list_all_active_conversations",
            [{"id": 1, "title": "Private conversation"}],
            "conversations_*.json",
        ),
        (
            "_export_notes_worker",
            "list_notes",
            [
                {
                    "id": 1,
                    "title": "Private note",
                    "created_at": "2026-07-23",
                    "updated_at": "2026-07-23",
                    "content": "private body",
                }
            ],
            "notes_*/*.md",
        ),
        (
            "_export_characters_worker",
            "list_character_cards",
            [{"id": 1, "name": "Private character"}],
            "characters_*.json",
        ),
    ],
)
def test_settings_sensitive_exports_are_private_under_umask_zero(
    tmp_path: Path,
    monkeypatch,
    worker_name: str,
    result_method: str,
    records: list[dict],
    file_pattern: str,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    export_root = home / ".local" / "share" / "tldw_cli" / "exports"
    export_root.mkdir(parents=True, mode=0o755)
    database = tmp_path / "chachanotes.db"
    _create_private_database(database)
    target = _worker_target(database)
    mock_database = MagicMock()
    getattr(mock_database, result_method).return_value = records
    monkeypatch.setattr(
        settings_module,
        "CharactersRAGDB",
        MagicMock(return_value=mock_database),
    )

    previous = os.umask(0)
    try:
        worker = getattr(settings_module.ToolsSettingsWindow, worker_name)
        worker(target)
    finally:
        os.umask(previous)

    exported = list(export_root.glob(file_pattern))
    assert len(exported) == 1
    assert export_root.stat().st_mode & 0o777 == 0o700
    assert exported[0].parent.stat().st_mode & 0o777 == 0o700
    assert exported[0].stat().st_mode & 0o777 == 0o600
    mock_database.close_connection.assert_called_once_with()


@pytest.mark.skipif(os.name != "posix", reason="POSIX restore privacy contract")
def test_settings_restore_uses_guarded_lifecycle_without_chmodding_custom_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_parent = tmp_path / "custom"
    custom_parent.mkdir(mode=0o755)
    database = custom_parent / "media.db"
    selected_backup = tmp_path / "selected-backup.db"
    _create_private_database(database)
    _create_private_database(selected_backup)
    live = sqlite3.connect(database)
    live.execute("UPDATE settings_probe SET value = 'before'")
    live.execute("PRAGMA journal_mode=WAL")
    live.commit()
    live.close()
    selected = sqlite3.connect(selected_backup)
    selected.execute("UPDATE settings_probe SET value = 'after'")
    selected.commit()
    selected.close()
    custom_parent.chmod(0o755)
    target = _worker_target(database)
    calls: list[tuple[str, str, Path, Path, Path]] = []

    def tracking_restore(
        owner_id,
        pre_restore_owner_id,
        source,
        destination,
        pre_restore,
    ):
        calls.append(
            (
                owner_id,
                pre_restore_owner_id,
                Path(source),
                Path(destination),
                Path(pre_restore),
            )
        )
        return restore_private_sqlite(
            owner_id,
            pre_restore_owner_id,
            source,
            destination,
            pre_restore,
        )

    monkeypatch.setattr(
        settings_module,
        "restore_private_sqlite",
        tracking_restore,
        raising=False,
    )

    settings_module.ToolsSettingsWindow._restore_single_worker.__wrapped__(
        target,
        "media",
        selected_backup,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call[:4] == (
        "settings.restore",
        "settings.pre_restore_backup",
        selected_backup,
        database,
    )
    assert call[4].parent == custom_parent
    assert (call[4].stat().st_mode & 0o777) == 0o600
    assert custom_parent.stat().st_mode & 0o777 == 0o755
    restored = sqlite3.connect(database)
    try:
        assert (
            restored.execute("SELECT value FROM settings_probe").fetchone()[0]
            == "after"
        )
    finally:
        restored.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX restore lock contract")
def test_settings_restore_busy_never_reports_success_or_changes_live_database(
    tmp_path: Path,
) -> None:
    database = tmp_path / "media.db"
    selected_backup = tmp_path / "selected-backup.db"
    _create_private_database(database)
    _create_private_database(selected_backup)
    live = sqlite3.connect(database)
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("UPDATE settings_probe SET value = 'before'")
    live.commit()
    selected = sqlite3.connect(selected_backup)
    selected.execute("UPDATE settings_probe SET value = 'after'")
    selected.commit()
    selected.close()
    live.execute("SELECT value FROM settings_probe").fetchone()
    assert live.in_transaction is False
    target = _worker_target(database)
    try:
        settings_module.ToolsSettingsWindow._restore_single_worker.__wrapped__(
            target,
            "media",
            selected_backup,
        )
    finally:
        live.close()

    notifications = [str(call) for call in target.app_instance.notify.call_args_list]
    assert any("live restore is unavailable" in call for call in notifications)
    assert not any("restored successfully" in call for call in notifications)
    verification = sqlite3.connect(database)
    try:
        assert (
            verification.execute("SELECT value FROM settings_probe").fetchone()[0]
            == "before"
        )
    finally:
        verification.close()
    assert list(tmp_path.glob("media_pre_restore_*.db")) == []


def test_settings_restore_indeterminate_state_warns_against_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    database = tmp_path / "media.db"
    selected_backup = tmp_path / "selected-backup.db"
    _create_private_database(database)
    _create_private_database(selected_backup)
    target = _worker_target(database)
    pre_restore = tmp_path / "media_pre_restore_test.db"

    def indeterminate_restore(*_args, **_kwargs):
        raise private_sqlite.SQLiteRestoreIndeterminateError(
            database,
            pre_restore,
        )

    monkeypatch.setattr(
        settings_module,
        "restore_private_sqlite",
        indeterminate_restore,
    )

    settings_module.ToolsSettingsWindow._restore_single_worker.__wrapped__(
        target,
        "media",
        selected_backup,
    )

    notifications = [str(call) for call in target.app_instance.notify.call_args_list]
    assert any("may already contain restored data" in call for call in notifications)
    assert any(str(pre_restore) in call for call in notifications)
    assert any("Do not retry" in call for call in notifications)
    assert not any("restored successfully" in call for call in notifications)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX restore directory contract")
async def test_settings_restore_picker_directory_is_private_under_umask_zero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    target = _worker_target(tmp_path / "unused.db")

    async def dismiss_picker(*args, **kwargs):
        return None

    target.app_instance.push_screen = dismiss_picker
    previous = os.umask(0)
    try:
        await settings_module.ToolsSettingsWindow._restore_single_database(
            target,
            "media",
        )
    finally:
        os.umask(previous)

    restore_dir = home / ".local" / "share" / "tldw_cli" / "backups" / "media"
    assert (restore_dir.stat().st_mode & 0o777) == 0o700
