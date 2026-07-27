"""Profile-store configuration and application ownership integration tests.

Task 10 extends this module with Backup All orchestration coverage. Task 9
deliberately covers only path resolution and ownership construction.
"""

from __future__ import annotations

import asyncio
import ast
import json
import shutil
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from loguru import logger
from textual.app import App, ComposeResult
from textual.worker import WorkerCancelled, WorkerFailed

from tldw_chatbook import config
import tldw_chatbook.UI.Tools_Settings_Window as tools_settings_module
from tldw_chatbook.UI.Tools_Settings_Window import ToolsSettingsWindow


REPO_ROOT = Path(__file__).resolve().parents[2]


class _RecordingProfileRepository:
    """Small online-backup fake with explicit ordering controls."""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.error = error
        self.release = release
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.destinations: list[Path] = []

    async def backup_to(self, destination: Path) -> None:
        self.destinations.append(destination)
        self.started.set()
        try:
            if self.release is not None:
                await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.error is not None:
            raise self.error

        with sqlite3.connect(destination) as connection:
            connection.execute(
                "CREATE TABLE profile_backup_probe (id INTEGER PRIMARY KEY)"
            )


class _BackupApp:
    """App boundary fake used to verify repository ownership and notices."""

    def __init__(
        self,
        repository: _RecordingProfileRepository | None,
    ) -> None:
        self.repository = repository
        self.ensure_calls = 0
        self.notifications: list[tuple[str, dict[str, Any]]] = []

    async def _ensure_tts_profile_repository(
        self,
    ) -> _RecordingProfileRepository | None:
        self.ensure_calls += 1
        return self.repository

    def notify(self, message: str, **kwargs: Any) -> None:
        self.notifications.append((message, kwargs))


class _WorkerLike:
    """Non-awaitable stand-in for Textual's Worker contract."""

    def __init__(
        self,
        work: Callable[[], Any],
        *,
        before_run: Callable[[], Awaitable[None]] | None = None,
        observed_errors: list[Exception] | None = None,
    ) -> None:
        self._work = work
        self._before_run = before_run
        self._observed_errors = observed_errors
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    async def wait(self) -> Any:
        try:
            if self._before_run is not None:
                await self._before_run()
            return await asyncio.to_thread(self._work)
        except asyncio.CancelledError as error:
            raise WorkerCancelled("Worker was cancelled") from error
        except Exception as error:
            if self._observed_errors is not None:
                self._observed_errors.append(error)
            raise WorkerFailed(error) from None


class _RealBackupWindow(ToolsSettingsWindow):
    """Minimal mounted settings window for real Textual worker tests."""

    def compose(self) -> ComposeResult:
        yield from ()


class _RealBackupApp(App[None]):
    """Real Textual app boundary for worker lifecycle regression coverage."""

    def __init__(
        self,
        *,
        repository: _RecordingProfileRepository | None,
        config_data: dict[str, Any],
    ) -> None:
        super().__init__()
        self.repository = repository
        self.ensure_calls = 0
        self.notifications: list[tuple[str, dict[str, Any]]] = []
        self.settings_window = _RealBackupWindow(app_instance=self)
        self.settings_window.config_data = config_data

    def compose(self) -> ComposeResult:
        yield self.settings_window

    async def _ensure_tts_profile_repository(
        self,
    ) -> _RecordingProfileRepository | None:
        self.ensure_calls += 1
        return self.repository

    def notify(self, message: str, **kwargs: Any) -> None:
        self.notifications.append((message, kwargs))


def _prepare_real_backup_app(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repository: _RecordingProfileRepository | None,
) -> tuple[_RealBackupApp, tuple[Path, Path, Path]]:
    """Create a mounted-app candidate with isolated legacy database paths."""

    chachanotes_path = tmp_path / "legacy-chachanotes.db"
    prompts_path = tmp_path / "legacy-prompts.db"
    media_path = tmp_path / "legacy-media.db"
    for source in (chachanotes_path, prompts_path, media_path):
        source.write_bytes(b"database")

    monkeypatch.setattr(
        tools_settings_module.Path,
        "home",
        classmethod(lambda cls: tmp_path),
    )
    monkeypatch.setattr(
        tools_settings_module,
        "get_prompts_db_path",
        lambda: prompts_path,
    )
    monkeypatch.setattr(
        tools_settings_module,
        "load_cli_config_and_ensure_existence",
        lambda: {},
    )

    app = _RealBackupApp(
        repository=repository,
        config_data={
            "database": {
                "chachanotes_db_path": str(chachanotes_path),
                "media_db_path": str(media_path),
            }
        },
    )
    return app, (chachanotes_path, prompts_path, media_path)


def _prepare_backup_window(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app: _BackupApp,
    run_worker: Callable[..., Any],
) -> tuple[ToolsSettingsWindow, Path, list[Path]]:
    chachanotes_path = tmp_path / "legacy-chachanotes.db"
    prompts_path = tmp_path / "legacy-prompts.db"
    media_path = tmp_path / "legacy-media.db"
    profile_path = tmp_path / "private-profile-store.db"
    for source in (chachanotes_path, prompts_path, media_path, profile_path):
        source.write_bytes(b"database")

    monkeypatch.setattr(
        tools_settings_module.Path,
        "home",
        classmethod(lambda cls: tmp_path),
    )
    monkeypatch.setattr(
        tools_settings_module,
        "get_prompts_db_path",
        lambda: prompts_path,
    )

    copy_sources: list[Path] = []
    real_copy2 = shutil.copy2

    def recording_copy2(source: Path, destination: Path, *args: Any, **kwargs: Any):
        copy_sources.append(Path(source))
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr(tools_settings_module.shutil, "copy2", recording_copy2)

    window = object.__new__(ToolsSettingsWindow)
    window._app_instance = app
    window.config_data = {
        "database": {
            "chachanotes_db_path": str(chachanotes_path),
            "media_db_path": str(media_path),
            "tts_profiles_db_path": str(profile_path),
        }
    }
    window.run_worker = run_worker
    return window, profile_path, copy_sources


def _terminal_notifications(app: _BackupApp) -> list[tuple[str, dict[str, Any]]]:
    return [
        notification
        for notification in app.notifications
        if notification[1].get("severity") in {"success", "warning", "error"}
    ]


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


@pytest.mark.asyncio
async def test_real_worker_cancellation_before_legacy_publication_leaves_no_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _RecordingProfileRepository()
    app, _ = _prepare_real_backup_app(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        repository=repository,
    )
    copy_started = threading.Event()
    release_copy = threading.Event()
    copy_finished = threading.Event()
    real_copy2 = shutil.copy2

    def blocking_copy2(
        source: Path,
        destination: Path,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        result = real_copy2(source, destination, *args, **kwargs)
        copy_started.set()
        release_copy.wait(timeout=2)
        copy_finished.set()
        return result

    monkeypatch.setattr(tools_settings_module.shutil, "copy2", blocking_copy2)

    async with app.run_test() as pilot:
        backup_task = asyncio.create_task(app.settings_window._backup_databases())
        await asyncio.wait_for(asyncio.to_thread(copy_started.wait), timeout=1)
        backup_task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await backup_task
        finally:
            release_copy.set()

        await asyncio.wait_for(asyncio.to_thread(copy_finished.wait), timeout=1)
        await pilot.pause()

    backup_root = tmp_path / ".local" / "share" / "tldw_cli" / "backups"
    assert tuple(backup_root.rglob("*.db")) == ()
    assert tuple(backup_root.rglob("*.tmp")) == ()
    assert tuple(backup_root.iterdir()) == ()
    assert _terminal_notifications(app) == []
    assert app.settings_window._backup_all_in_progress is False


@pytest.mark.asyncio
async def test_real_worker_cancellation_before_manifest_publication_leaves_no_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _RecordingProfileRepository()
    app, _ = _prepare_real_backup_app(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        repository=repository,
    )
    dump_started = threading.Event()
    release_dump = threading.Event()
    dump_finished = threading.Event()
    real_dump = json.dump

    def blocking_dump(*args: Any, **kwargs: Any) -> None:
        real_dump(*args, **kwargs)
        dump_started.set()
        release_dump.wait(timeout=2)
        dump_finished.set()

    monkeypatch.setattr(tools_settings_module.json, "dump", blocking_dump)

    async with app.run_test() as pilot:
        backup_task = asyncio.create_task(app.settings_window._backup_databases())
        await asyncio.wait_for(asyncio.to_thread(dump_started.wait), timeout=1)
        backup_task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await backup_task
        finally:
            release_dump.set()

        await asyncio.wait_for(asyncio.to_thread(dump_finished.wait), timeout=1)
        await pilot.pause()

    backup_root = tmp_path / ".local" / "share" / "tldw_cli" / "backups"
    assert tuple(backup_root.rglob("backup_info.json")) == ()
    assert tuple(backup_root.rglob("*.tmp")) == ()
    assert _terminal_notifications(app) == []
    assert app.settings_window._backup_all_in_progress is False


def test_manifest_mid_write_failure_preserves_previous_file_and_cleans_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_dir = tmp_path / "backup"
    backup_dir.mkdir()
    manifest_path = backup_dir / "backup_info.json"
    previous_manifest = {"timestamp": "previous", "databases": []}
    manifest_path.write_text(json.dumps(previous_manifest), encoding="utf-8")

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
            backup_dir,
            (),
        )

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == previous_manifest
    assert tuple(backup_dir.glob("*.tmp")) == ()
    assert tuple(backup_dir.glob(".*.tmp")) == ()


def test_manifest_cleanup_failure_does_not_mask_base_exception_or_expose_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backup_dir = tmp_path / "private-backup-directory"
    backup_dir.mkdir()
    private_error = f"cleanup failed at {backup_dir / 'private-stage.tmp'}"

    def interrupt_mid_dump(
        value: Any,
        stream: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        stream.write('{"timestamp": "partial')
        raise KeyboardInterrupt("original manifest interruption")

    real_unlink = Path.unlink

    def fail_stage_cleanup(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if path.parent == backup_dir and path.suffix == ".tmp":
            raise RuntimeError(private_error)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tools_settings_module.json, "dump", interrupt_mid_dump)
    monkeypatch.setattr(tools_settings_module.Path, "unlink", fail_stage_cleanup)
    log_messages: list[str] = []
    sink_id = logger.add(log_messages.append, level="WARNING", format="{message}")

    try:
        with pytest.raises(KeyboardInterrupt, match="original manifest interruption"):
            ToolsSettingsWindow._write_backup_manifest(
                "20260727_010203",
                backup_dir,
                (),
            )
    finally:
        logger.remove(sink_id)
        for temporary_path in backup_dir.iterdir():
            real_unlink(temporary_path)

    public_copy = "\n".join(map(str, log_messages))
    assert "cleanup=unlink failed" in public_copy
    assert private_error not in public_copy
    assert str(backup_dir) not in public_copy


@pytest.mark.asyncio
async def test_real_backup_all_same_clock_uses_distinct_timestamp_prefixed_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixed_timestamp = "20260727_010203"

    class FixedDatetime:
        @classmethod
        def now(cls) -> FixedDatetime:
            return cls()

        def strftime(self, date_format: str) -> str:
            assert date_format == "%Y%m%d_%H%M%S"
            return fixed_timestamp

    monkeypatch.setattr(tools_settings_module, "datetime", FixedDatetime)
    repository = _RecordingProfileRepository()
    app, _ = _prepare_real_backup_app(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        repository=repository,
    )

    async with app.run_test():
        await app.settings_window._backup_databases()
        await app.settings_window._backup_databases()

    backup_root = tmp_path / ".local" / "share" / "tldw_cli" / "backups"
    backup_directories = tuple(
        sorted(path for path in backup_root.iterdir() if path.is_dir())
    )
    assert len(backup_directories) == 2
    assert all(
        backup_dir.name.startswith(f"{fixed_timestamp}_")
        for backup_dir in backup_directories
    )
    assert {path.parent for path in repository.destinations} == set(backup_directories)
    assert all((path / "backup_info.json").exists() for path in backup_directories)
    assert app.settings_window._backup_all_in_progress is False


@pytest.mark.asyncio
async def test_real_backup_all_rejects_duplicate_orchestration_while_profile_waits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release_profile = asyncio.Event()
    repository = _RecordingProfileRepository(release=release_profile)
    app, _ = _prepare_real_backup_app(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        repository=repository,
    )

    async with app.run_test() as pilot:
        first_backup = asyncio.create_task(app.settings_window._backup_databases())
        await asyncio.wait_for(repository.started.wait(), timeout=1)
        duplicate_backup = asyncio.create_task(app.settings_window._backup_databases())

        try:
            for _ in range(20):
                await pilot.pause()
                if duplicate_backup.done() or len(repository.destinations) > 1:
                    break
            assert duplicate_backup.done()
            assert len(repository.destinations) == 1
        finally:
            release_profile.set()
            await asyncio.gather(first_backup, duplicate_backup)

    assert (
        "Database backup is already in progress.",
        {"severity": "warning"},
    ) in app.notifications
    assert app.settings_window._backup_all_in_progress is False


@pytest.mark.asyncio
async def test_real_backup_workers_expose_only_value_free_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = _RecordingProfileRepository()
    app, source_paths = _prepare_real_backup_app(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        repository=repository,
    )
    configured_profile_path = tmp_path / "private-configured-profile-store.db"
    app.settings_window.config_data["database"]["tts_profiles_db_path"] = str(
        configured_profile_path
    )
    workers: list[Any] = []

    async with app.run_test():
        real_run_worker = app.settings_window.run_worker

        def recording_run_worker(work: Any, **kwargs: Any) -> Any:
            worker = real_run_worker(work, **kwargs)
            workers.append(worker)
            return worker

        monkeypatch.setattr(
            app.settings_window,
            "run_worker",
            recording_run_worker,
        )
        await app.settings_window._backup_databases()

    assert [worker.description for worker in workers] == [
        "Copy legacy database backups",
        "Write database backup manifest",
    ]
    sensitive_values = {
        *(str(path) for path in source_paths),
        str(configured_profile_path),
        *(str(path) for path in repository.destinations),
        *(str(path.parent) for path in repository.destinations),
    }
    for worker in workers:
        public_metadata = "\n".join(
            (worker.name, worker.group, worker.description, repr(worker))
        )
        assert all(value not in public_metadata for value in sensitive_values)


@pytest.mark.asyncio
async def test_backup_all_awaits_profile_and_manifest_before_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release_profile = asyncio.Event()
    repository = _RecordingProfileRepository(release=release_profile)
    app = _BackupApp(repository)
    worker_calls: list[dict[str, Any]] = []
    manifest_started = asyncio.Event()
    release_manifest = asyncio.Event()

    def run_worker(
        work: Callable[[], Any],
        **kwargs: Any,
    ) -> _WorkerLike:
        worker_calls.append(kwargs)

        async def pause_manifest() -> None:
            manifest_started.set()
            await release_manifest.wait()

        before_run = pause_manifest if len(worker_calls) == 2 else None
        return _WorkerLike(work, before_run=before_run)

    window, profile_path, copy_sources = _prepare_backup_window(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        app=app,
        run_worker=run_worker,
    )

    backup_task = asyncio.create_task(ToolsSettingsWindow._backup_databases(window))
    await asyncio.wait_for(repository.started.wait(), timeout=1)
    assert _terminal_notifications(app) == []

    release_profile.set()
    await asyncio.wait_for(manifest_started.wait(), timeout=1)
    assert _terminal_notifications(app) == []

    destination = repository.destinations[0]
    manifest_path = destination.parent / "backup_info.json"
    assert not manifest_path.exists()

    release_manifest.set()
    await asyncio.wait_for(backup_task, timeout=1)

    assert app.ensure_calls == 1
    assert repository.destinations == [destination]
    assert destination.parent.parent == (
        tmp_path / ".local" / "share" / "tldw_cli" / "backups"
    )
    assert destination.parent.name
    assert destination.exists()
    with sqlite3.connect(destination) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)

    backup_info = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["name"] for entry in backup_info["databases"]] == [
        "ChaChaNotes",
        "Prompts",
        "Media",
        "TTS Profiles",
    ]
    assert backup_info["databases"][-1]["path"] == str(destination)
    assert profile_path not in copy_sources
    assert worker_calls == [
        {
            "name": "backup_worker",
            "group": "tts_profile_backup_all",
            "description": "Copy legacy database backups",
            "thread": True,
            "exclusive": True,
            "exit_on_error": False,
        },
        {
            "name": "backup_manifest_worker",
            "group": "tts_profile_backup_all",
            "description": "Write database backup manifest",
            "thread": True,
            "exclusive": True,
            "exit_on_error": False,
        },
    ]

    terminal = _terminal_notifications(app)
    assert len(terminal) == 1
    assert terminal[0][1]["severity"] == "success"
    assert str(destination.parent) not in terminal[0][0]
    assert window._backup_all_in_progress is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ("unavailable", "backup_failure"))
async def test_backup_all_records_only_legacy_entries_on_profile_partial_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_mode: str,
) -> None:
    private_error = f"profile backup exploded at {tmp_path / 'private-store.db'}"
    repository = (
        None
        if failure_mode == "unavailable"
        else _RecordingProfileRepository(error=RuntimeError(private_error))
    )
    app = _BackupApp(repository)
    worker_calls: list[dict[str, Any]] = []

    def run_worker(
        work: Callable[[], Any],
        **kwargs: Any,
    ) -> _WorkerLike:
        worker_calls.append(kwargs)
        return _WorkerLike(work)

    window, profile_path, copy_sources = _prepare_backup_window(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        app=app,
        run_worker=run_worker,
    )
    log_messages: list[str] = []
    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")

    try:
        await ToolsSettingsWindow._backup_databases(window)
    finally:
        logger.remove(sink_id)

    backup_root = tmp_path / ".local" / "share" / "tldw_cli" / "backups"
    backup_directories = tuple(backup_root.iterdir())
    assert len(backup_directories) == 1
    manifest_path = backup_directories[0] / "backup_info.json"
    backup_info = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert [entry["name"] for entry in backup_info["databases"]] == [
        "ChaChaNotes",
        "Prompts",
        "Media",
    ]

    assert app.ensure_calls == 1
    if repository is not None:
        assert len(repository.destinations) == 1
    assert profile_path not in copy_sources
    assert len(worker_calls) == 2
    assert all(
        call
        == {
            "name": expected_name,
            "group": "tts_profile_backup_all",
            "description": expected_description,
            "thread": True,
            "exclusive": True,
            "exit_on_error": False,
        }
        for call, (expected_name, expected_description) in zip(
            worker_calls,
            (
                ("backup_worker", "Copy legacy database backups"),
                (
                    "backup_manifest_worker",
                    "Write database backup manifest",
                ),
            ),
            strict=True,
        )
    )

    terminal = _terminal_notifications(app)
    assert len(terminal) == 1
    assert terminal[0][1]["severity"] != "success"
    assert "partial" in terminal[0][0].lower()
    public_copy = "\n".join(
        [message for message, _ in app.notifications] + log_messages
    )
    assert private_error not in public_copy
    assert str(profile_path) not in public_copy
    assert window._backup_all_in_progress is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ("legacy", "manifest"))
async def test_backup_all_worker_failure_is_private_and_never_reports_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_phase: str,
) -> None:
    private_error = f"worker exploded at {tmp_path / 'private-worker.db'}"
    repository = _RecordingProfileRepository()
    app = _BackupApp(repository)
    worker_calls: list[dict[str, Any]] = []
    workers: list[_WorkerLike] = []
    observed_errors: list[Exception] = []

    def run_worker(
        work: Callable[[], Any],
        **kwargs: Any,
    ) -> _WorkerLike:
        worker_calls.append(kwargs)
        worker = _WorkerLike(work, observed_errors=observed_errors)
        workers.append(worker)
        return worker

    window, profile_path, _ = _prepare_backup_window(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        app=app,
        run_worker=run_worker,
    )

    def fail_worker_io(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(private_error)

    if failure_phase == "legacy":
        monkeypatch.setattr(
            tools_settings_module.shutil,
            "copy2",
            fail_worker_io,
        )
    else:
        monkeypatch.setattr(
            tools_settings_module.json,
            "dump",
            fail_worker_io,
        )
    log_messages: list[str] = []
    sink_id = logger.add(log_messages.append, level="DEBUG", format="{message}")

    try:
        await ToolsSettingsWindow._backup_databases(window)
    finally:
        logger.remove(sink_id)

    terminal = _terminal_notifications(app)
    assert terminal == [("Database backup failed.", {"severity": "error"})]
    assert all(kwargs.get("severity") != "success" for _, kwargs in app.notifications)
    public_copy = "\n".join(
        [message for message, _ in app.notifications] + log_messages
    )
    assert private_error not in public_copy
    assert str(profile_path) not in public_copy
    assert app.ensure_calls == (0 if failure_phase == "legacy" else 1)
    assert len(worker_calls) == (1 if failure_phase == "legacy" else 2)
    assert len(observed_errors) == 1
    assert private_error not in repr(observed_errors[0])
    assert workers[-1].cancelled
    assert not any(worker.cancelled for worker in workers[:-1])
    assert window._backup_all_in_progress is False
    if failure_phase == "legacy":
        backup_root = tmp_path / ".local" / "share" / "tldw_cli" / "backups"
        assert tuple(backup_root.iterdir()) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", ("legacy", "profile", "manifest"))
async def test_backup_all_cancellation_does_not_write_manifest_or_notify_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cancel_phase: str,
) -> None:
    release_profile = asyncio.Event() if cancel_phase == "profile" else None
    repository = _RecordingProfileRepository(release=release_profile)
    app = _BackupApp(repository)
    worker_started = asyncio.Event()
    release_worker = asyncio.Event()
    worker_calls = 0
    workers: list[_WorkerLike] = []

    def run_worker(
        work: Callable[[], Any],
        **kwargs: Any,
    ) -> _WorkerLike:
        nonlocal worker_calls
        worker_calls += 1
        expected_call = 1 if cancel_phase == "legacy" else 2

        async def pause_worker() -> None:
            worker_started.set()
            await release_worker.wait()

        before_run = pause_worker if worker_calls == expected_call else None
        worker = _WorkerLike(work, before_run=before_run)
        workers.append(worker)
        return worker

    window, _, _ = _prepare_backup_window(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        app=app,
        run_worker=run_worker,
    )
    backup_task = asyncio.create_task(ToolsSettingsWindow._backup_databases(window))
    started = (
        repository.started.wait()
        if cancel_phase == "profile"
        else worker_started.wait()
    )
    await asyncio.wait_for(started, timeout=1)

    backup_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await backup_task

    if cancel_phase == "profile":
        await asyncio.wait_for(repository.cancelled.wait(), timeout=1)
        assert not any(worker.cancelled for worker in workers)
    else:
        assert workers[-1].cancelled
        assert not any(worker.cancelled for worker in workers[:-1])
    assert _terminal_notifications(app) == []
    assert tuple(tmp_path.rglob("backup_info.json")) == ()
    assert window._backup_all_in_progress is False
