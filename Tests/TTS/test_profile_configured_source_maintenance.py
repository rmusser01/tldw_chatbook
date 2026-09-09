"""Configured TTS source and delegated real SQLite pin ownership."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _delegated_pin_child(root, after, final_pin=True):
    import asyncio
    import os
    import time
    import types

    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    native_os = private.os
    private.os = types.SimpleNamespace(**vars(native_os))
    original_open = private._open_artifact_fd
    original_native_close = private.private_paths._native_close
    selected = []
    prepared = False
    original_prepare = private._prepare_source_artifacts

    def prepare(*args, **kwargs):
        nonlocal prepared
        result = original_prepare(*args, **kwargs)
        prepared = True
        return result

    private._prepare_source_artifacts = prepare
    calls = []
    retired_parents = []

    def open_artifact(parent_fd, leaf, **kwargs):
        fd = original_open(parent_fd, leaf, **kwargs)
        if (
            leaf == "profiles.sqlite"
            and not kwargs["writable"]
            and (prepared or not final_pin)
        ):
            selected.append((parent_fd, fd))
        return fd

    def close(fd):
        if selected and fd == selected[0][1]:
            calls.append(fd)
            if after:
                native_os.close(fd)
            raise OSError("delegated source pin close uncertainty")
        result = native_os.close(fd)
        if selected and fd == selected[0][0]:
            with pytest.raises(OSError):
                native_os.fstat(fd)
            retired_parents.append(fd)
        return result

    def native_close(fd):
        result = original_native_close(fd)
        if selected and fd == selected[0][0]:
            with pytest.raises(OSError):
                native_os.fstat(fd)
            retired_parents.append(fd)
        return result

    private.private_paths._native_close = native_close
    private._open_artifact_fd = open_artifact
    private.os.close = close
    with pytest.raises(ProfileRepositoryError):
        await repo.backup_to(root / "snapshot.sqlite")
    parent_fd, file_fd = selected[0]
    assert len(calls) == 1
    if after:
        with pytest.raises(OSError):
            os.fstat(file_fd)
    else:
        assert os.fstat(file_fd).st_ino == (root / "profiles.sqlite").stat().st_ino
    if hasattr(private, "_SQLiteSourcePinJob"):
        job = next(
            j for j in storage._raw_operations if type(j) is private._SQLiteSourcePinJob
        )
        if final_pin:
            assert job.source.file_fd == file_fd and job.source.parent_fd == -1
            assert job.body_error is None
        else:
            assert any(
                r[0] == file_fd and r[2] and not r[3] for r in job.preflight_descriptors
            )
        assert parent_fd in retired_parents
        assert job.cleanup_errors
        assert all(lease in storage._live_leases for lease in job.leases)
    await repo.close()
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.02), (
            "actual delegated native pin uncertainty was lost after outer backup and repo close"
        )
    finally:
        pause.resume()


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("final_pin", [False, True])
def test_delegated_backup_pin_uncertainty_survives_repository_close(
    tmp_path, after, final_pin
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_configured_source_maintenance import _delegated_pin_child
asyncio.run(_delegated_pin_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(after),
        str(final_pin),
    )


async def _configured_app_child(root, change):
    import importlib
    import asyncio
    import copy
    import threading
    import sys
    import types

    config_path = root / "home" / "config.toml"
    config_path.write_text(
        '[general]\nuser_folder="default_user"\n'
        f'base_data_dir="{root / "data"}"\n'
        "[database]\n"
        f'tts_profiles_db_path="{root / "configured.sqlite"}"\n'
        "[splash_screen]\nenabled=false\n"
        "[model_catalog]\nenabled=false\n"
    )
    config_path.chmod(0o600)
    import tldw_chatbook.config as config
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    app = TldwCli()
    repo = app._tts_profile_repository
    try:
        assert await app._ensure_tts_profile_repository() is repo
        assert repo._configured_source.repository is repo
        if change in ("harmless", "cache_replacement"):
            if change == "cache_replacement":
                config._CONFIG_CACHE = copy.deepcopy(config._CONFIG_CACHE)
            config._CONFIG_CACHE.setdefault("appearance", {})["theme"] = "new-theme"
            assert (await repo.list_profiles()).value.profiles == ()
            assert await app._ensure_tts_profile_repository() is repo
            return
        if change == "custom":
            from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

            custom = TTSProfileRepository(root / "custom.sqlite")
            assert custom._configured_source is None
            try:
                await custom.open()
                assert (await custom.list_profiles()).value.profiles == ()
            finally:
                await custom.close()
            return
        if change in ("copy", "subclass_copy"):
            if change == "copy":
                copied = copy.copy(repo)
            else:

                class Subclass(type(repo)):
                    pass

                copied = object.__new__(Subclass)
                copied.__dict__.update(repo.__dict__)
            with pytest.raises(ProfileRepositoryError):
                await copied.list_profiles()
            assert (await repo.list_profiles()).value.profiles == ()
            return
        if change == "queued":
            entered, finish = threading.Event(), threading.Event()

            def hold_worker():
                entered.set()
                assert finish.wait(5)

            held = repo._executor.submit(hold_worker)
            assert await asyncio.to_thread(entered.wait, 3)
            pending = asyncio.create_task(repo.list_profiles())
            await asyncio.sleep(0)
            config._CONFIG_CACHE["database"]["tts_profiles_db_path"] = str(
                root / "foreign.sqlite"
            )
            finish.set()
            await asyncio.wrap_future(held)
            with pytest.raises(ProfileRepositoryError):
                await pending
            assert not repo._pending_futures and not repo._publication_completions
            return
        if change == "callable_equality":
            calls = []

            class Replacement:
                def __eq__(self, other):
                    calls.append("equality")
                    return True

                def __call__(self):
                    calls.append("selector")
                    return config_path

            config._get_effective_config_path = Replacement()
            with pytest.raises(ProfileRepositoryError):
                await repo.list_profiles()
            assert calls == []
            return
        if change == "path":
            config._CONFIG_CACHE["database"]["tts_profiles_db_path"] = str(
                root / "foreign.sqlite"
            )
        elif change == "cache_source":
            config._CONFIG_CACHE_SOURCE = root / "foreign.toml"
        elif change == "module":
            replacement = types.ModuleType(config.__name__)
            replacement.__dict__.update(vars(config))
            sys.modules[config.__name__] = replacement
        elif change == "repository_path":
            repo._database_path = root / "foreign.sqlite"
        elif change == "repository_reload":
            import tldw_chatbook.TTS.profile_repository as repository_module

            importlib.reload(repository_module)
        else:
            config._CONFIG_CACHE["general"]["users_name"] = "foreign_user"
        with pytest.raises(ProfileRepositoryError):
            await repo.list_profiles()
    finally:
        await app._close_tts_profile_repository()
    assert not (root / "foreign.sqlite").exists()


@pytest.mark.parametrize(
    "change",
    [
        "path",
        "cache_source",
        "module",
        "repository_path",
        "repository_reload",
        "profile",
        "harmless",
        "cache_replacement",
        "custom",
        "copy",
        "subclass_copy",
        "queued",
        "callable_equality",
    ],
)
def test_actual_app_configured_repository_refuses_changed_config_source(
    tmp_path, change
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_configured_source_maintenance import _configured_app_child
asyncio.run(_configured_app_child(Path(sys.argv[1]), sys.argv[2]))
""",
        change,
    )


async def _refused_completed_child(root, cancelled):
    import asyncio
    import sqlite3
    import threading
    import time

    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    native = repo._connection
    executor = repo._executor
    hold = next(iter(storage._holds.values()))
    before = (root / "profiles.sqlite").read_bytes()
    entered, finish = threading.Event(), threading.Event()
    validation = repo._worker_validate_standalone_snapshot

    def blocked(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return validation(*args, **kwargs)

    repo._worker_validate_standalone_snapshot = blocked
    pending = asyncio.create_task(repo.backup_to(root / "snapshot.sqlite"))
    assert await asyncio.to_thread(entered.wait, 3)
    workers = tuple(repo._pending_futures)
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    try:
        assert not await repo._maintenance_drain(time.monotonic() + 0.02)
        if cancelled:
            pending.cancel()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        finish.set()
        if not cancelled:
            with pytest.raises(ProfileRepositoryError):
                await pending
        else:
            for worker in workers:
                with pytest.raises(ProfileRepositoryError):
                    await asyncio.wrap_future(worker)
        assert await repo._maintenance_drain(time.monotonic() + 3)
        assert pause.drain(time.monotonic() + 0.05)
        assert not repo._pending_futures and not repo._publication_completions
        assert not repo._backup_native_operations
        assert not (root / "snapshot.sqlite").exists()
        assert not tuple(root.glob(".snapshot.sqlite.*.backup"))
        assert (root / "profiles.sqlite").read_bytes() == before

        def native_closed():
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                native.execute("SELECT 1")

        await asyncio.wrap_future(executor.submit(native_closed))
        storage._shutdown()  # Known private repository-only child; no app/runtime release.
        assert _probe(hold.authority.control_root, hold.names) == "entered"
    finally:
        finish.set()
        pause.resume()
        await repo.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_refused_disposable_backup_completes_actual_local_and_native_boundary(
    tmp_path, cancelled
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_configured_source_maintenance import _refused_completed_child
asyncio.run(_refused_completed_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(cancelled),
    )
