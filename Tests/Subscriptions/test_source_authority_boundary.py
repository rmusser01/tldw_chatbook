"""Only explicit app input can mint stored Watchlists URL authority."""

import json

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Utils.egress import UrlProvenance, provenance_of_stored_source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim, expected",
    [
        (None, UrlProvenance.UNKNOWN),
        ("user_entered", UrlProvenance.UNKNOWN),
        (UrlProvenance.USER_ENTERED, UrlProvenance.USER_ENTERED),
    ],
)
async def test_shared_writer_rejects_serialized_authority_claim(
    tmp_path, monkeypatch, claim, expected
):
    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default=None: default,
    )
    db = SubscriptionsDB(tmp_path / "subscriptions.db", "authority-test")
    try:
        service = LocalWatchlistsService(db_factory=lambda: db)
        row = await service.create_source(
            {
                "name": "test",
                "source_type": "url",
                "url": "http://intranet/a",
                "url_provenance": claim,
                "processing_options": {
                    "url_provenance": "user_entered",
                    "max_items": 5,
                },
            }
        )
        stored = db.get_subscription(row["source_id"])
        assert provenance_of_stored_source(stored["processing_options"]) is expected
        assert json.loads(stored["processing_options"])["max_items"] == 5
        await service.update_source(row["source_id"], {"url": "http://intranet/b"})
        assert (
            provenance_of_stored_source(
                db.get_subscription(row["source_id"])["processing_options"]
            )
            is UrlProvenance.UNKNOWN
        )
        await service.update_source(
            row["source_id"],
            {"url": "http://intranet/c", "url_provenance": UrlProvenance.USER_ENTERED},
        )
        await service.update_source(row["source_id"], {"active": False})
        assert (
            provenance_of_stored_source(
                db.get_subscription(row["source_id"])["processing_options"]
            )
            is UrlProvenance.USER_ENTERED
        )
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored, expected",
    [
        (None, "unknown"),
        ("{", "unknown"),
        ('{"url_provenance":"bogus", "max_items":5}', "unknown"),
        ('{"url_provenance":"user_entered", "max_items":5}', "user_entered"),
    ],
)
@pytest.mark.parametrize("migration_owner", ["runtime", "recovery"])
async def test_restored_v2_source_has_conservative_lineage_before_first_fetch(
    tmp_path, monkeypatch, stored, expected, migration_owner
):
    import sqlite3

    import httpx

    from Tests.Subscriptions.test_url_fetch_source_provenance import _URL
    from tldw_chatbook.Subscriptions.monitoring_engine import (
        FetchBlockedError,
        URLMonitor,
    )
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(
        "tldw_chatbook.DB.Subscriptions_DB.get_cli_setting",
        lambda section, key, default=None: default,
    )
    monkeypatch.setattr(
        egress, "get_cli_setting", lambda section, key, default=None: default
    )
    path = tmp_path / "subscriptions.db"
    db = SubscriptionsDB(path, "authority-test")
    source_id = db.add_subscription("legacy", "url", _URL)
    db.close()
    # v2 and v3 have the same physical columns; v3's migration owns the data.
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_version SET version=2")
        connection.execute("UPDATE subscriptions SET processing_options=?", (stored,))
    with sqlite3.connect(path) as connection:
        trigger = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE name='update_subscriptions_timestamp'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER update_subscriptions_timestamp")
        connection.execute("UPDATE subscriptions SET updated_at='2000-01-01 00:00:00'")
        connection.execute(trigger)
        before_updated = connection.execute(
            "SELECT updated_at FROM subscriptions"
        ).fetchone()[0]
    if migration_owner == "recovery":
        from threading import Event

        from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
        from tldw_chatbook.DB.recovery_operations import recovery_adapters

        adapter = next(
            a for a in recovery_adapters() if a.owner_id == "db.subscriptions"
        )
        from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

        denied = []
        original = _Restrictions.authorize

        def audit(self, *args):
            result = original(self, *args)
            if result == sqlite3.SQLITE_DENY:
                denied.append(args)
            return result

        monkeypatch.setattr(_Restrictions, "authorize", audit)
        assert validate_candidate(adapter, path, Event(), migrate=True) == (), denied
    sent = []

    async def resolve(host):
        return ["192.168.1.1"]

    async def send(self, request, **kwargs):
        sent.append(str(request.url))
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/html"},
            content=b"<p>Price: 10</p>",
        )

    monkeypatch.setattr(egress, "_resolve_async", resolve)
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    for attempt in range(2):
        reopened = SubscriptionsDB(path, "authority-test")
        try:
            row = reopened.get_subscription(source_id)
            assert row["updated_at"] == before_updated == "2000-01-01 00:00:00"
            assert (
                reopened.conn.execute(
                    "SELECT sql FROM sqlite_schema WHERE name='update_subscriptions_timestamp'"
                ).fetchone()[0]
                == trigger
            )
            options = json.loads(row["processing_options"])
            assert options["url_provenance"] == expected
            if stored and "max_items" in stored:
                assert options["max_items"] == 5
            if expected == "unknown":
                with pytest.raises(FetchBlockedError):
                    await URLMonitor(reopened)._fetch_url_content(row)
                assert sent == []
            else:
                await URLMonitor(reopened)._fetch_url_content(row)
                assert len(sent) == attempt + 1
        finally:
            reopened.close()


def test_subscriptions_provenance_migration_failure_rolls_back_data_trigger_and_version(
    tmp_path, monkeypatch
):
    import sqlite3
    from pathlib import Path

    path = tmp_path / "subscriptions.db"
    original = SubscriptionsDB(path, "rollback-test")
    original.add_subscription(
        "legacy", "url", "http://intranet/a", auto_pause_threshold=10
    )
    original.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_version SET version=2")
        connection.execute("UPDATE subscriptions SET processing_options=NULL")
        before_rows = connection.execute("SELECT * FROM subscriptions").fetchall()
        before_schema = connection.execute(
            "SELECT type,name,sql FROM sqlite_schema ORDER BY type,name"
        ).fetchall()
    read_text = Path.read_text

    def broken_migration(file, *args, **kwargs):
        content = read_text(file, *args, **kwargs)
        if file.name == "subscriptions_v2_to_v3_url_provenance.sql":
            return content + "\nTHIS IS AN INJECTED SQL FAULT;\n"
        return content

    monkeypatch.setattr(Path, "read_text", broken_migration)
    failed = SubscriptionsDB.__new__(SubscriptionsDB)
    try:
        with pytest.raises(sqlite3.OperationalError):
            failed.__init__(path, "rollback-test")
    finally:
        failed.close()
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT version FROM schema_version").fetchall() == [
            (2,)
        ]
        assert (
            connection.execute("SELECT * FROM subscriptions").fetchall() == before_rows
        )
        assert (
            connection.execute(
                "SELECT type,name,sql FROM sqlite_schema ORDER BY type,name"
            ).fetchall()
            == before_schema
        )
    monkeypatch.setattr(Path, "read_text", read_text)
    reopened = SubscriptionsDB(path, "rollback-test")
    try:
        assert (
            json.loads(reopened.get_subscription(1)["processing_options"])[
                "url_provenance"
            ]
            == "unknown"
        )
    finally:
        reopened.close()
