"""Time-based ``sync_log`` retention (task-34389, ADR-229).

The reachability retention from task-19564 bounds ``sync_log`` to the rows
its readers can reach, but the frontier rows of live entities still live
forever, and the media DB journal has no reachability bound at all -- the
prune APIs shipped with zero production callers. task-34389 adds the missing
half: a time-based window (default 90 days, ``[sync]
sync_log_retention_days``, ``0`` disables) swept by one parameterized
``DELETE`` per store, wired into the app's post-startup maintenance path
(next to change-review retention).

What this pins:

* Both stores' ``delete_sync_log_entries_older_than`` delete ONLY rows
  strictly older than the cutoff, including rows written by the real
  trigger/``_log_sync_event`` paths (so the ISO ``YYYY-MM-DDTHH:MM:SS.mmmZ``
  timestamp shapes are what actually get compared).
* A row at exactly the cutoff survives (strictly-older semantics).
* The app-level pass reads the config knob, prunes BOTH stores in one pass,
  and ``0`` -- the documented off-switch -- touches neither.
* The shipped config carries the knob and a fresh profile resolves its
  default through the real loader.
"""

from __future__ import annotations

import asyncio
import tomllib
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

OLD_TS = "2020-01-01T00:00:00.000Z"


def _cutoff(days: int = 90) -> str:
    """The cutoff string the app pass computes (mirrors app.py)."""
    return (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "sync_retention.db", "retention_client")
    try:
        yield database
    finally:
        database.close_connection()


@pytest.fixture()
def media_db(tmp_path: Path) -> MediaDatabase:
    # File-backed on purpose: the app pass runs the DELETE on a worker
    # thread, and MediaDatabase keeps thread-local connections, so an
    # in-memory DB would be a different (empty) database over there --
    # production DBs are file-backed, and so is this fixture.
    database = MediaDatabase(db_path=str(tmp_path / "media_sync_retention.db"), client_id="retention_client")
    try:
        yield database
    finally:
        database.close_connection()


def _media_log_event(db: MediaDatabase, entity_uuid: str, version: int = 1) -> None:
    """Write one sync event through the real production logging path."""
    with db.transaction() as conn:
        db._log_sync_event(
            conn,
            entity="Media",
            entity_uuid=entity_uuid,
            operation="create",
            version=version,
            payload={"uuid": entity_uuid, "content": "payload body"},
        )


# ---------------------------------------------------------------------------
# ChaChaNotes_DB.delete_sync_log_entries_older_than
# ---------------------------------------------------------------------------
def test_chachanotes_prune_deletes_only_rows_older_than_cutoff(
    chacha_db: CharactersRAGDB,
):
    """Rows backdated past the window leave; fresh trigger-written rows stay."""
    cutoff = _cutoff()
    conversation_id = chacha_db.add_conversation(
        {"title": "retention probe", "character_id": 1}
    )
    fresh_message_id = chacha_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "fresh body",
        }
    )

    total_rows = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log"
    ).fetchone()["n"]
    assert total_rows > 0

    # Backdate everything EXCEPT the fresh message's rows, via direct SQL.
    chacha_db.execute_query(
        "UPDATE sync_log SET timestamp = ? WHERE NOT (entity = 'messages' "
        "AND entity_id = ?)",
        (OLD_TS, fresh_message_id),
        commit=True,
    )
    backdated = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
        (cutoff,),
    ).fetchone()["n"]
    assert backdated > 0

    deleted = chacha_db.delete_sync_log_entries_older_than(cutoff)

    assert deleted == backdated
    # Compare in SQL: the DATETIME column comes back as a Python datetime.
    stale_left = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
        (cutoff,),
    ).fetchone()["n"]
    assert stale_left == 0
    remaining = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log"
    ).fetchone()["n"]
    assert remaining > 0, "the fresh message's sync rows must survive"


def test_chachanotes_prune_keeps_row_at_exact_cutoff(chacha_db: CharactersRAGDB):
    """Strictly-older semantics: a row stamped exactly AT the cutoff survives."""
    cutoff = _cutoff()
    with chacha_db.transaction() as conn:
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES('messages', 'm-boundary', "
            "'create', ?, 'retention_client', 1, '{}')",
            (cutoff,),
        )
        conn.execute(
            "INSERT INTO sync_log(entity, entity_id, operation, timestamp, "
            "client_id, version, payload) VALUES('messages', 'm-ancient', "
            "'create', ?, 'retention_client', 1, '{}')",
            (OLD_TS,),
        )

    deleted = chacha_db.delete_sync_log_entries_older_than(cutoff)

    assert deleted == 1
    survivors = {
        row["entity_id"]
        for row in chacha_db.execute_query(
            "SELECT entity_id FROM sync_log"
        ).fetchall()
    }
    # Membership, not an exact set: a fresh profile seeds its own (current-
    # timestamp) rows, which legitimately survive.
    assert "m-boundary" in survivors
    assert "m-ancient" not in survivors


def test_chachanotes_prune_writes_still_land_after_prune(
    chacha_db: CharactersRAGDB,
):
    """Pruning is a pure DELETE: triggers keep emitting afterwards."""
    chacha_db.delete_sync_log_entries_older_than(OLD_TS)
    conversation_id = chacha_db.add_conversation(
        {"title": "post-prune", "character_id": 1}
    )
    hits = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log WHERE entity = 'conversations' "
        "AND entity_id = ?",
        (conversation_id,),
    ).fetchone()["n"]
    assert hits >= 1


@pytest.mark.parametrize("bad_cutoff", ["", None, 123])
def test_chachanotes_prune_rejects_invalid_cutoff(
    chacha_db: CharactersRAGDB, bad_cutoff
):
    with pytest.raises(ValueError):
        chacha_db.delete_sync_log_entries_older_than(bad_cutoff)


# ---------------------------------------------------------------------------
# Client_Media_DB_v2.delete_sync_log_entries_older_than
# ---------------------------------------------------------------------------
def test_media_prune_deletes_only_rows_older_than_cutoff(media_db: MediaDatabase):
    """Events backdated past the window leave; fresh ones stay."""
    for i in range(3):
        _media_log_event(media_db, f"uuid-{i}")

    media_db.execute_query(
        "UPDATE sync_log SET timestamp = ? WHERE entity_uuid IN ('uuid-0', 'uuid-1')",
        (OLD_TS,),
        commit=True,
    )

    deleted = media_db.delete_sync_log_entries_older_than(_cutoff())

    assert deleted == 2
    survivors = {
        row["entity_uuid"]
        for row in media_db.execute_query(
            "SELECT entity_uuid FROM sync_log"
        ).fetchall()
    }
    assert survivors == {"uuid-2"}


def test_media_prune_keeps_row_at_exact_cutoff(media_db: MediaDatabase):
    cutoff = _cutoff()
    media_db.execute_query(
        "INSERT INTO sync_log(entity, entity_uuid, operation, timestamp, "
        "client_id, version, payload) VALUES('Media', 'uuid-boundary', "
        "'create', ?, 'retention_client', 1, '{}')",
        (cutoff,),
        commit=True,
    )

    assert media_db.delete_sync_log_entries_older_than(cutoff) == 0
    assert (
        media_db.execute_query(
            "SELECT COUNT(*) AS n FROM sync_log"
        ).fetchone()["n"]
        == 1
    )


@pytest.mark.parametrize("bad_cutoff", ["", None, 123])
def test_media_prune_rejects_invalid_cutoff(media_db: MediaDatabase, bad_cutoff):
    with pytest.raises(ValueError):
        media_db.delete_sync_log_entries_older_than(bad_cutoff)


# ---------------------------------------------------------------------------
# The app-level pass: TldwCli._perform_sync_log_retention
# ---------------------------------------------------------------------------
def _fake_app(**dbs) -> SimpleNamespace:
    return SimpleNamespace(loguru_logger=loguru_logger, **dbs)


def _run_pass(monkeypatch: pytest.MonkeyPatch, fake_app: SimpleNamespace, days) -> None:
    """Drive LifecycleMixin._perform_sync_log_retention without booting the app."""
    from tldw_chatbook import app_lifecycle as lifecycle_module

    # Patch the name the code actually reads: the pass lives on LifecycleMixin
    # and resolves get_cli_setting in app_lifecycle's namespace (the perf
    # snapshot carried a seam pointing at tldw_chatbook.app, the pre-port
    # home -- patching there silently missed, so the "0 disables" case ran a
    # real 90-day prune and failed).
    monkeypatch.setattr(
        lifecycle_module,
        "get_cli_setting",
        lambda *args, **kwargs: days,
    )
    asyncio.run(
        lifecycle_module.LifecycleMixin._perform_sync_log_retention(fake_app)
    )


def test_app_pass_prunes_both_stores(
    chacha_db: CharactersRAGDB,
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """One pass prunes both journals; nothing under the window is touched."""
    chacha_db.add_conversation({"title": "app pass", "character_id": 1})
    chacha_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )
    _media_log_event(media_db, f"uuid-{uuid.uuid4().hex[:8]}")
    media_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )

    _run_pass(monkeypatch, _fake_app(chachanotes_db=chacha_db, media_db=media_db), 90)

    # The contract: NOTHING older than the window survives in either store.
    # (A fresh profile's per-thread connection may legitimately seed new
    # current-timestamp rows during the pass; those are not the target.)
    for db in (chacha_db, media_db):
        stale_left = db.execute_query(
            "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
            (_cutoff(),),
        ).fetchone()["n"]
        assert stale_left == 0


def test_app_pass_zero_disables(
    chacha_db: CharactersRAGDB,
    media_db: MediaDatabase,
    monkeypatch: pytest.MonkeyPatch,
):
    """``[sync] sync_log_retention_days = 0`` is the documented off-switch."""
    chacha_db.add_conversation({"title": "keep everything", "character_id": 1})
    chacha_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )
    _media_log_event(media_db, f"uuid-{uuid.uuid4().hex[:8]}")
    media_db.execute_query(
        "UPDATE sync_log SET timestamp = ?", (OLD_TS,), commit=True
    )

    _run_pass(monkeypatch, _fake_app(chachanotes_db=chacha_db, media_db=media_db), 0)

    # Ancient rows are all still there: the pass never ran a DELETE.
    for db in (chacha_db, media_db):
        stale_left = db.execute_query(
            "SELECT COUNT(*) AS n FROM sync_log WHERE timestamp < ?",
            (_cutoff(),),
        ).fetchone()["n"]
        assert stale_left > 0


def test_app_pass_skips_missing_dbs(monkeypatch: pytest.MonkeyPatch):
    """A store that is not open (media_db is per-host, may be None) is skipped."""

    class RecordingDB:
        def __init__(self):
            self.cutoffs: list[str] = []

        def delete_sync_log_entries_older_than(self, cutoff: str) -> int:
            self.cutoffs.append(cutoff)
            return 5

    recording = RecordingDB()
    _run_pass(monkeypatch, _fake_app(chachanotes_db=recording, media_db=None), 90)
    assert len(recording.cutoffs) == 1
    # The cutoff is the ISO-Z shape both journals store.
    assert recording.cutoffs[0].endswith("Z")
    assert "T" in recording.cutoffs[0]


def test_app_pass_invalid_config_value_is_contained(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    """A garbage knob value logs and skips; it never raises out of the pass."""
    chacha_db.add_conversation({"title": "invalid knob", "character_id": 1})
    before = chacha_db.execute_query(
        "SELECT COUNT(*) AS n FROM sync_log"
    ).fetchone()["n"]

    _run_pass(
        monkeypatch,
        _fake_app(chachanotes_db=chacha_db, media_db=None),
        "not-a-number",
    )

    assert (
        chacha_db.execute_query("SELECT COUNT(*) AS n FROM sync_log").fetchone()["n"]
        == before
    )


# ---------------------------------------------------------------------------
# The shipped config carries the knob (Tests/test_config_chunking_defaults shape)
# ---------------------------------------------------------------------------
def test_config_template_ships_sync_section():
    from tldw_chatbook import config as config_module

    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert template["sync"]["sync_log_retention_days"] == 90


def test_default_config_tree_carries_sync_section():
    from tldw_chatbook import config as config_module

    assert config_module.DEFAULT_CONFIG_FROM_TOML["sync"]["sync_log_retention_days"] == 90


def test_real_loader_resolves_sync_retention_default(tmp_path, monkeypatch):
    """A fresh profile (missing config file) still resolves the knob."""
    from Tests.Backup_Recovery.config_test_support import select_config_source

    # Select the fresh profile through the ADR-126-compliant seam: a raw
    # TLDW_CONFIG_PATH redirect fails the config-participant admission closed
    # (RecoveryRequired) because the imported source stays bound to the
    # bootstrap selection. select_config_source installs an independent
    # source module, exactly as Tests/test_config_chunking_defaults.py does.
    source = select_config_source(monkeypatch, tmp_path / "missing-config.toml")

    settings = source.load_settings(force_reload=True)

    assert settings["sync"]["sync_log_retention_days"] == 90
    assert source.get_cli_setting("sync", "sync_log_retention_days", None) == 90
