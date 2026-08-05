"""
Tests for notes file-watcher UI notification coalescing (TASK-1352).

A burst of watchdog filesystem events must collapse into at most one
UI-facing ``on_files_changed`` notification per coalescing window, instead
of one callback (previously one asyncio task) per raw event.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tldw_chatbook.Notes.auto_sync_manager import (
    FILES_CHANGED_NOTIFY_COALESCE_SECONDS,
    AutoSyncManager,
    NotesFileWatcher,
)


def _make_manager(tmp_path: Path) -> AutoSyncManager:
    manager = AutoSyncManager(
        sync_service=MagicMock(),
        sync_folder=tmp_path,
        user_id="test-user",
    )
    # Simulate start() without spinning up the watchdog observer/sync loop.
    manager.is_running = True
    manager._loop = asyncio.get_running_loop()
    manager.file_watcher = NotesFileWatcher(manager._on_file_changed)
    return manager


def _fire_modify(manager: AutoSyncManager, path: Path) -> None:
    """Simulate a watchdog on_modified event through the watcher."""
    event = MagicMock()
    event.is_directory = False
    event.src_path = str(path)
    manager.file_watcher.on_modified(event)


@pytest.mark.asyncio
async def test_event_burst_produces_single_ui_notification(tmp_path):
    manager = _make_manager(tmp_path)
    notifications = []
    manager.on_files_changed = notifications.append

    for i in range(10):
        _fire_modify(manager, tmp_path / f"note-{i}.md")

    await asyncio.sleep(FILES_CHANGED_NOTIFY_COALESCE_SECONDS * 3)

    assert notifications == [10]
    assert manager.pending_sync


@pytest.mark.asyncio
async def test_separate_bursts_each_produce_one_notification(tmp_path):
    manager = _make_manager(tmp_path)
    notifications = []
    manager.on_files_changed = notifications.append

    _fire_modify(manager, tmp_path / "a.md")
    _fire_modify(manager, tmp_path / "b.md")
    await asyncio.sleep(FILES_CHANGED_NOTIFY_COALESCE_SECONDS * 3)

    _fire_modify(manager, tmp_path / "c.md")
    await asyncio.sleep(FILES_CHANGED_NOTIFY_COALESCE_SECONDS * 3)

    assert len(notifications) == 2


@pytest.mark.asyncio
async def test_stop_cancels_pending_notification(tmp_path):
    manager = _make_manager(tmp_path)
    notifications = []
    manager.on_files_changed = notifications.append

    _fire_modify(manager, tmp_path / "note.md")
    manager.stop()

    await asyncio.sleep(FILES_CHANGED_NOTIFY_COALESCE_SECONDS * 3)
    assert notifications == []


@pytest.mark.asyncio
async def test_no_callback_set_still_marks_pending_sync(tmp_path):
    manager = _make_manager(tmp_path)
    manager._on_file_changed(tmp_path / "note.md")
    assert manager.pending_sync
