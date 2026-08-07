"""Tests for DB-size telemetry (task-2859 item 5: WAL-inclusive sizes).

Covers ``tldw_chatbook.Utils.Utils.get_formatted_db_size_with_wal`` (the
stat helper) and ``DBStatusManager.update_db_sizes`` (the seam that feeds
the Library rail's Details disclosure via ``app.db_sizes_status``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

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
