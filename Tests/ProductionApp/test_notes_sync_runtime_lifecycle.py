"""Production-app ownership proof for the gated lasting-sync runtime."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass

import pytest

from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncStoreSetting,
)
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.app_factory import _build_test_app


pytestmark = pytest.mark.unit


@dataclass
class _RuntimeProbe:
    events: list[str]
    starts: int = 0
    shutdowns: int = 0

    async def start(self) -> None:
        self.starts += 1
        self.events.append("sync-started")

    async def shutdown(self) -> None:
        self.shutdowns += 1
        self.events.append("sync-stopped")


@dataclass
class _FileOwnerProbe:
    events: list[str]

    async def shutdown_async(self) -> None:
        self.events.append("file-notes-stopped")


def test_app_constructs_exactly_one_cutover_runtime_after_notes_scope_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    runtime = _RuntimeProbe([])

    def build(**kwargs: object) -> _RuntimeProbe:
        calls.append(kwargs)
        return runtime

    # TASK-21108: the builders are imported function-locally inside
    # `TldwCli._construct_notes_sync_runtime_owner`, so the substitution has
    # to land on the defining module; `app_module` no longer holds them.
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "build_notes_sync_runtime_owner", build)

    app = _build_test_app(configured_default="chat")

    # TASK-21108: construction is deferred to first access, not done in
    # `__init__` -- that deferral is what keeps the 15-module lasting-sync
    # chain off the app import path.
    assert calls == []
    assert app.notes_sync_runtime_owner is runtime
    assert len(calls) == 1
    assert calls[0]["notes_scope_service"] is app.notes_scope_service
    assert calls[0]["cutover_admitted"] is True
    assert calls[0]["profile_process_is_sole"] is app._instance_lock_status.acquired


@pytest.mark.asyncio
@pytest.mark.parametrize("marker_present", [False, True])
async def test_real_mounted_runtime_migrates_then_opens_the_cutover_gate(
    marker_present: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_chatbook.app as app_module

    migrations: list[str] = []
    # TASK-21108: patch the defining module -- app.py imports this
    # function-locally now (see _construct_notes_sync_runtime_owner).
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "build_notes_sync_legacy_migrator",
        lambda **_kwargs: lambda: migrations.append("migrated"),
    )
    store = NotesDeviceStateStore(app_module.get_notes_sync_state_db_path())
    store.initialize()
    if marker_present:
        store.set_setting(
            NotesSyncStoreSetting("cutover_marker", "notes-sync-cutover-v1")
        )
    app = _build_test_app(configured_default="chat")

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app._notes_sync_runtime_start_task
        owner = app.notes_sync_runtime_owner
        assert type(owner) is NotesSyncRuntimeOwner
        assert owner.snapshot().status == "active"
        assert owner._coordinator is not None
        assert owner._watcher is None
        assert owner._leases == {}

    assert migrations == ([] if marker_present else ["migrated"])


@pytest.mark.asyncio
async def test_runtime_start_completion_refreshes_the_current_library_screen() -> None:
    from types import SimpleNamespace

    from tldw_chatbook.app import TldwCli

    refreshed: list[str] = []
    screen = SimpleNamespace(
        refresh_notes_sync_runtime=lambda: refreshed.append("refreshed")
    )
    app = SimpleNamespace(
        screen=screen,
        call_after_refresh=lambda callback: callback(),
    )
    task = asyncio.create_task(asyncio.sleep(0))
    await task

    TldwCli._observe_notes_sync_runtime_start(app, task)

    assert refreshed == ["refreshed"]


@pytest.mark.asyncio
async def test_mounted_migration_failure_is_bounded_and_observed() -> None:
    import tldw_chatbook.app as app_module

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        # TASK-21112: the boot gate defers an unconfigured start, so give it
        # on-disk evidence first — this test is about a migration failure
        # inside an admitted start, not about the gate.
        NotesDeviceStateStore(app_module.get_notes_sync_state_db_path()).initialize()
        app = _build_test_app(configured_default="chat")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await app._notes_sync_runtime_start_task
            assert (
                app.notes_sync_runtime_owner.snapshot().status,
                app.notes_sync_runtime_owner.snapshot().next_action,
            ) == ("failed", "review_settings")
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert unhandled == []


@pytest.mark.asyncio
async def test_runtime_starts_without_library_and_shuts_down_once() -> None:
    events: list[str] = []
    app = _build_test_app(configured_default="chat")
    runtime = _RuntimeProbe(events)
    app.notes_sync_runtime_owner = runtime

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, LibraryScreen)

    assert runtime.starts == 1
    assert runtime.shutdowns == 1


@pytest.mark.asyncio
async def test_sync_runtime_shutdown_precedes_file_notes_and_generic_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    app = _build_test_app(configured_default="chat")
    app.notes_sync_runtime_owner = _RuntimeProbe(events)
    app._notes_sync_runtime_shutdown_task = None
    app.file_notes_session_owner = _FileOwnerProbe(events)
    app._file_notes_session_owner_shutdown_task = None
    app._audio_cpp_artifact_lease_coordinator = None

    async def no_op() -> None:
        events.append("generic-owner-stopped")

    monkeypatch.setattr(app.audio_cpp_model_install_owner, "shutdown", no_op)
    monkeypatch.setattr(app, "_shutdown_console_image_edits", no_op)
    monkeypatch.setattr(app, "_shutdown_console_runtime", no_op)

    await app._shutdown_app_owned_lifecycles()

    assert events.index("sync-stopped") < events.index("file-notes-stopped")
    assert events.index("sync-stopped") < events.index("generic-owner-stopped")


def test_library_screen_does_not_construct_or_own_sync_runtime() -> None:
    source = inspect.getsource(LibraryScreen)

    assert "NotesSyncRuntimeOwner" not in source
    assert "build_notes_sync_runtime_owner" not in source


@pytest.mark.asyncio
async def test_zero_profile_boot_creates_no_notes_sync_state_db(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-21112 pin: a boot that never uses notes sync creates NO state DB.

    The runtime used to start unconditionally: every boot created the state
    database, ran its schema census, and the first boot ran the legacy
    evidence SELECTs over chachanotes.db. The gate must leave the filesystem
    untouched — through shutdown as well.
    """

    import tldw_chatbook.app as app_module

    state_path = tmp_path / "notes_sync_state.db"
    monkeypatch.setattr(
        app_module, "get_notes_sync_state_db_path", lambda: state_path
    )
    app = _build_test_app(configured_default="chat")

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await app._notes_sync_runtime_start_task
        owner = app.notes_sync_runtime_owner
        assert type(owner) is NotesSyncRuntimeOwner
        assert (owner.snapshot().status, owner.snapshot().next_action) == (
            "not_configured",
            "none",
        )
        assert not state_path.exists()

    # The shutdown path must not create it either.
    assert not state_path.exists()
    assert app.notes_sync_runtime_owner.snapshot().status == "stopped"


@pytest.mark.asyncio
async def test_legacy_sync_directory_key_still_boots_the_migration_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-time legacy migration must still run for users with the key."""

    import tldw_chatbook.app as app_module

    state_path = tmp_path / "notes_sync_state.db"
    monkeypatch.setattr(
        app_module, "get_notes_sync_state_db_path", lambda: state_path
    )
    migrations: list[str] = []
    # TASK-21108: patch the defining module -- app.py imports this
    # function-locally now (see _construct_notes_sync_runtime_owner).
    import tldw_chatbook.Notes.notes_sync_runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "build_notes_sync_legacy_migrator",
        lambda **_kwargs: lambda: migrations.append("migrated"),
    )
    app = _build_test_app(
        configured_default="chat",
        config_overrides={"notes": {"sync_directory": str(tmp_path / "legacy")}},
    )

    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await app._notes_sync_runtime_start_task
            owner = app.notes_sync_runtime_owner
            assert owner.snapshot().status == "active"
            assert migrations == ["migrated"]
            assert state_path.exists()
    finally:
        # config_overrides deep-merge into load_settings()'s cached dict;
        # scrub the legacy key so it cannot leak into later tests.
        notes = app.app_config.get("notes")
        if isinstance(notes, dict):
            notes.pop("sync_directory", None)
