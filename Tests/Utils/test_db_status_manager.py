"""Tests for DB-size telemetry (task-2859 item 5: WAL-inclusive sizes).

Covers ``tldw_chatbook.Utils.Utils.get_formatted_db_size_with_wal`` (the
stat helper) and ``DBStatusManager.update_db_sizes`` (the seam that feeds
the Library rail's Details disclosure via ``app.db_sizes_status``).
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_chatbook.Utils.db_status_manager import DBStatusManager
from tldw_chatbook.Utils.Utils import (
    get_formatted_db_size_with_wal,
    get_formatted_file_size,
)


# --- get_formatted_file_size (unchanged behavior after the refactor) --------


def test_get_formatted_file_size_missing_file_returns_none(tmp_path: Path):
    assert get_formatted_file_size(tmp_path / "missing.db") is None


def test_get_formatted_file_size_formats_kilobytes(tmp_path: Path):
    db_path = tmp_path / "small.db"
    db_path.write_bytes(b"x" * 2048)
    assert get_formatted_file_size(db_path) == "2.0 KB"


# --- get_formatted_db_size_with_wal ------------------------------------------


def test_wal_size_missing_main_file_returns_none(tmp_path: Path):
    assert get_formatted_db_size_with_wal(tmp_path / "missing.db") is None


def test_wal_size_matches_plain_size_when_no_sidecars_exist(tmp_path: Path):
    """No ``-wal``/``-shm`` sidecars (the common post-checkpoint case): the
    WAL-inclusive helper must agree exactly with the plain formatter."""
    db_path = tmp_path / "checkpointed.db"
    db_path.write_bytes(b"x" * 4096)
    assert get_formatted_db_size_with_wal(db_path) == get_formatted_file_size(db_path)


def test_wal_size_includes_the_wal_sidecar_the_main_file_alone_misses():
    """task-2859 item 5's reported case: a small main file (reads "4.0 KB"
    alone) with a WAL sidecar holding megabytes of uncheckpointed writes --
    the combined report must reflect the real footprint, not just the main
    file."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "ledger.db"
        db_path.write_bytes(b"x" * 4096)  # 4.0 KB main file
        wal_path = Path(tmp) / "ledger.db-wal"
        wal_path.write_bytes(b"x" * (4 * 1024 * 1024))  # 4 MB WAL

        plain = get_formatted_file_size(db_path)
        combined = get_formatted_db_size_with_wal(db_path)

        assert plain == "4.0 KB"
        assert combined == "4.0 MB"
        assert combined != plain


def test_wal_size_includes_both_wal_and_shm_sidecars():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "ledger.db"
        db_path.write_bytes(b"x" * 1024)
        (Path(tmp) / "ledger.db-wal").write_bytes(b"x" * 1024)
        (Path(tmp) / "ledger.db-shm").write_bytes(b"x" * 1024)

        # 1024 (main) + 1024 (wal) + 1024 (shm) = 3072 bytes = 3.0 KB
        assert get_formatted_db_size_with_wal(db_path) == "3.0 KB"


# --- DBStatusManager.update_db_sizes: uses the WAL-inclusive helper ---------


@pytest.mark.asyncio
async def test_update_db_sizes_uses_the_wal_inclusive_helper(tmp_path, monkeypatch):
    prompts_db = tmp_path / "prompts.db"
    prompts_db.write_bytes(b"x" * 1024)
    (tmp_path / "prompts.db-wal").write_bytes(b"x" * (2 * 1024 * 1024))

    chachanotes_db = tmp_path / "chacha.db"
    chachanotes_db.write_bytes(b"x" * 1024)

    media_db = tmp_path / "media.db"
    media_db.write_bytes(b"x" * 1024)

    monkeypatch.setattr(
        "tldw_chatbook.config.get_prompts_db_path", lambda: prompts_db
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.get_chachanotes_db_path", lambda: chachanotes_db
    )
    monkeypatch.setattr("tldw_chatbook.config.get_media_db_path", lambda: media_db)

    app = SimpleNamespace()
    manager = DBStatusManager(app)
    await manager.update_db_sizes()

    assert app.db_sizes_status["prompts"] == "2.0 MB"
    assert app.db_sizes_status["chachanotes"] == "1.0 KB"
    assert app.db_sizes_status["media"] == "1.0 KB"


# --- task-22220: collection off the event loop; change-gated log line --------


def _stage_three_dbs(tmp_path: Path, monkeypatch) -> Path:
    """Create three 1 KB DB files and point the config getters at them.

    Returns the prompts DB path (the one the change-probe grows).
    """
    prompts_db = tmp_path / "prompts.db"
    prompts_db.write_bytes(b"x" * 1024)
    chachanotes_db = tmp_path / "chacha.db"
    chachanotes_db.write_bytes(b"x" * 1024)
    media_db = tmp_path / "media.db"
    media_db.write_bytes(b"x" * 1024)

    monkeypatch.setattr("tldw_chatbook.config.get_prompts_db_path", lambda: prompts_db)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_chachanotes_db_path", lambda: chachanotes_db
    )
    monkeypatch.setattr("tldw_chatbook.config.get_media_db_path", lambda: media_db)
    return prompts_db


@pytest.mark.asyncio
async def test_db_size_stat_syscalls_run_off_the_event_loop(tmp_path, monkeypatch):
    """task-22220 item 1: every 120 s fire used to run ~15 stat/exists
    syscalls synchronously ON the event loop. Wrap ``os.stat`` (the syscall
    ``Path.exists``/``Path.is_file``/``os.path.getsize`` all bottom out in),
    scoped to the staged DB dir, and record which thread each call ran on:
    born red with every stat on the loop thread; green when the collection
    runs off-loop and zero stats hit the loop thread."""
    _stage_three_dbs(tmp_path, monkeypatch)

    loop_thread = threading.get_ident()
    stat_threads: list[int] = []
    real_stat = os.stat
    marker = str(tmp_path)

    def recording_stat(path, *args, **kwargs):
        try:
            text = os.fspath(path)
        except TypeError:
            text = ""
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        if marker in str(text):
            stat_threads.append(threading.get_ident())
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr("os.stat", recording_stat)

    manager = DBStatusManager(SimpleNamespace())
    await manager.update_db_sizes()

    assert stat_threads, "test premise: the collection stat'ed the staged DB files"
    on_loop = [ident for ident in stat_threads if ident == loop_thread]
    assert on_loop == [], (
        f"{len(on_loop)} of {len(stat_threads)} stat syscalls ran on the "
        "event-loop thread; the DB-size collection must run off-loop"
    )


@pytest.mark.asyncio
async def test_db_size_log_line_fires_only_on_change(tmp_path, monkeypatch):
    """task-22220 item 1: the INFO triple used to fire unconditionally every
    120 s. It must fire on the first collection (change from nothing) and on
    a real size change -- and stay silent for an unchanged fire. The
    ``db_sizes_status`` cache is still assigned every fire (the log is
    gated, not the telemetry)."""
    prompts_db = _stage_three_dbs(tmp_path, monkeypatch)

    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="INFO")
    try:
        app = SimpleNamespace()
        manager = DBStatusManager(app)

        await manager.update_db_sizes()  # first fire: sizes are news
        await manager.update_db_sizes()  # nothing changed on disk
        unchanged = [line for line in lines if "DB sizes:" in line]
        assert len(unchanged) == 1, (
            f"expected 1 'DB sizes:' log line across two unchanged fires, "
            f"got {len(unchanged)} (the periodic log must be change-gated)"
        )
        assert app.db_sizes_status["prompts"] == "1.0 KB"

        prompts_db.write_bytes(b"x" * (2 * 1024 * 1024))
        await manager.update_db_sizes()  # a size actually changed
        changed = [line for line in lines if "DB sizes:" in line]
        assert len(changed) == 2, "a real size change must still log"
        assert app.db_sizes_status["prompts"] == "2.0 MB"
    finally:
        logger.remove(sink_id)


@pytest.mark.asyncio
async def test_update_cancelled_mid_collection_writes_nothing(tmp_path, monkeypatch):
    """task-22220 teardown walk: a fire in flight while the app is shutting
    down gets cancelled at the off-loop await. Cancellation must propagate
    (not be swallowed by the catch-all) and no partial result may be
    published to the app."""
    _stage_three_dbs(tmp_path, monkeypatch)

    started = threading.Event()
    release = threading.Event()

    def blocking_formatter(path):
        started.set()
        release.wait(2)
        return "1.0 KB"

    monkeypatch.setattr(
        "tldw_chatbook.Utils.Utils.get_formatted_db_size_with_wal",
        blocking_formatter,
    )

    app = SimpleNamespace()
    manager = DBStatusManager(app)
    task = asyncio.create_task(manager.update_db_sizes())
    await asyncio.to_thread(started.wait, 2)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not hasattr(app, "db_sizes_status"), (
        "a cancelled collection must not publish a partial result"
    )
