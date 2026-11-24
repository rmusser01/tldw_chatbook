"""Exact standalone historical Subscription catalogs for real SQLite fixtures."""

import sqlite3
from contextlib import closing

from tldw_chatbook.DB.recovery_operations import _SUBSCRIPTIONS_SCHEMA


def historical_database(path, version=3, *, source_index=True):
    """Recreate the captured standalone source-index lineage, including native FTS."""
    schema = next(
        s
        for v, s in _SUBSCRIPTIONS_SCHEMA
        if v == version
        and any(sql.startswith("CREATE INDEX idx_subscriptions_source ") for sql in s)
        == source_index
        and not any("CREATE TABLE conversations " in sql for sql in s)
    )
    with closing(sqlite3.connect(path)) as con, con:
        for sql in schema:
            if sql.startswith(
                ("CREATE TABLE ", "CREATE VIRTUAL TABLE ")
            ) and not sql.startswith(
                (
                    "CREATE TABLE sqlite_sequence",
                    "CREATE TABLE 'subscription_items_fts_",
                )
            ):
                con.execute(sql)
        for sql in schema:
            if not sql.startswith(("CREATE TABLE ", "CREATE VIRTUAL TABLE ")):
                con.execute(sql)
        con.execute("INSERT INTO schema_version VALUES (?)", (version,))
        assert catalog(con) == schema


def catalog(con):
    return tuple(
        r[0]
        for r in con.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
        )
    )
