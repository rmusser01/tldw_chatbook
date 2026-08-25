"""A ChaChaNotes database upgrades itself, from any caller (task-21441).

The incident, not the rule. Schema v48 made the v47->v48 step raise
``SchemaError: Console library migration seed is required for v47 upgrade.``
unless the constructor had been handed a ``ConsoleLibraryMigrationSeed``, with
a fresh database exempted -- so the requirement bit exactly the upgrade case.
Every production construction site threads the seed, so the TUI was insulated
and no user saw an outage; sixteen tests were red, and the one that named the
mechanism honestly was ``Tests/Packaging/test_installed_distribution.py``,
which installs the wheel into an empty tree and opens a v35 database the way a
non-TUI consumer would. A migration step that requires caller-supplied data
turns "open the database" into "open the database from inside one
application", and nothing outside that application can upgrade a profile.

The seed is now optional and defaults to automatic retrieval OFF -- the same
value ``config.load_console_library_migration_seed`` yields for a missing key
and the same value the fresh-database path has always written. This module is
the guard that a future step cannot quietly reintroduce the requirement:
``test_a_bare_open_migrates...`` fails the moment any step in the chain needs
something the constructor did not get.

Evidence style follows ``test_chachanotes_v47_messages_fts_backfill.py``: the
interrupt is a real SIGKILL inside the migration transaction, the equivalence
claims are content HASHES over every table rather than row counts, and the
post-conditions are absolute rather than before/after snapshots.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from Tests.ChaChaNotesDB.historical_bootstrap import chachanotes_db_at_version
from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_NAME = CharactersRAGDB._SCHEMA_NAME
CURRENT = CharactersRAGDB._CURRENT_SCHEMA_VERSION

#: The only value the v47->v48 step generates that two identical runs cannot
#: agree on: `console_conversation_library_policy.updated_at` defaults to
#: CURRENT_TIMESTAMP. Masked in the content hash, and separately asserted to be
#: a real timestamp so masking cannot hide an empty column.
_VOLATILE = {("console_conversation_library_policy", "updated_at")}


def _raw_version(db_path: Path) -> int:
    """Read the stamped version WITHOUT the class that would migrate it."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as probe:
        return probe.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (SCHEMA_NAME,),
        ).fetchone()[0]


def _raw_table_names(db_path: Path) -> set[str]:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as probe:
        return {
            row[0]
            for row in probe.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }


def _content_hash(db_path: Path) -> str:
    """Hash the whole database: its schema text and every row of every table.

    Row order is normalized by sorting, so this is a content identity, not a
    storage-layout one -- two runs that produce the same rows agree even if
    SQLite laid the pages out differently.
    """
    digest = hashlib.sha256()
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as probe:
        schema = sorted(
            probe.execute(
                "SELECT type, name, tbl_name, COALESCE(sql, '') FROM sqlite_master"
            ).fetchall()
        )
        for entry in schema:
            digest.update(repr(entry).encode("utf-8"))
        tables = sorted(
            row[0]
            for row in probe.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        )
        for table in tables:
            columns = [
                row[1] for row in probe.execute(f'PRAGMA table_info("{table}")')
            ]
            rows = probe.execute(f'SELECT * FROM "{table}"').fetchall()
            masked = sorted(
                repr(
                    tuple(
                        "<volatile>" if (table, column) in _VOLATILE else value
                        for column, value in zip(columns, row)
                    )
                )
                for row in rows
            )
            digest.update(repr((table, columns, masked)).encode("utf-8"))
    return digest.hexdigest()


def _integrity_is_clean(db_path: Path) -> bool:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as probe:
        return probe.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def _policy_rows(db_path: Path) -> list[tuple]:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as probe:
        return probe.execute(
            "SELECT conversation_id, auto_retrieve_on_send, assistant_library_access,"
            " policy_revision, updated_at"
            " FROM console_conversation_library_policy ORDER BY conversation_id"
        ).fetchall()


def _trigger_sql(db_path: Path, name: str) -> str:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as probe:
        row = probe.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (name,),
        ).fetchone()
    assert row is not None, f"trigger {name} is missing"
    return " ".join(row[0].lower().split())


def _seed_historical(db_path: Path, version: int, titles: list[str]) -> list[str]:
    """Build a genuine ``version``-shaped database with ``titles`` conversations."""
    with chachanotes_db_at_version(db_path, version, client_id="t21441-seed") as db:
        conversation_ids = [
            db.add_conversation({"title": title, "character_id": 1})
            for title in titles
        ]
        for index, conversation_id in enumerate(conversation_ids):
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": f"body-{index:03d}",
                }
            )
    return conversation_ids


# ---------------------------------------------------------------------------
# AC #1 / AC #6 -- the database upgrades itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("start_version", [45, 47])
def test_a_bare_open_migrates_an_existing_database_to_current(
    tmp_path: Path, start_version: int
):
    """The regression guard: no constructor argument beyond a client id.

    Both entry points are covered because the requirement had two halves that
    fired on different databases: a pre-flight check keyed on entering at
    exactly v47, and the step's own check, which is what a v35/v45 database
    walking the whole chain actually hit.
    """
    db_path = tmp_path / "chachanotes.db"
    conversation_ids = _seed_historical(db_path, start_version, ["a", "b", "c"])
    assert _raw_version(db_path) == start_version

    upgraded = CharactersRAGDB(db_path, client_id="t21441-bare")
    try:
        assert upgraded.get_message_by_id is not None  # instance is usable
    finally:
        upgraded.close_connection()

    assert _raw_version(db_path) == CURRENT
    assert _integrity_is_clean(db_path)
    assert [row[0] for row in _policy_rows(db_path)] == sorted(conversation_ids)
    # Never / Allowed, the fail-safe default: no automatic retrieval.
    assert {(row[1], row[2], row[3]) for row in _policy_rows(db_path)} == {(0, 1, 1)}
    assert all(row[4] for row in _policy_rows(db_path))


def test_the_v47_and_v48_index_guarantees_survive_an_unseeded_upgrade(
    tmp_path: Path,
):
    """task-21100's membership guards and task-21128's UPDATE OF scope.

    Both are load-bearing and both live downstream of the step this task
    changed, so an unseeded upgrade has to land them exactly as a seeded one
    does.
    """
    db_path = tmp_path / "chachanotes.db"
    _seed_historical(db_path, 45, ["guarded"])
    CharactersRAGDB(db_path, client_id="t21441-guards").close_connection()

    messages_au = _trigger_sql(db_path, "messages_au")
    assert "after update of content, deleted on messages" in messages_au
    assert "messages_fts_docsize" in messages_au
    assert "old.deleted = 0" in messages_au
    assert "new.deleted = 0" in messages_au
    assert "messages_fts_docsize" in _trigger_sql(db_path, "messages_ad")


def test_an_unseeded_upgrade_is_content_identical_to_an_explicit_false_seed(
    tmp_path: Path,
):
    """The default is not a different outcome -- it is the same outcome.

    One fixture, copied, so every byte written before the migration is shared;
    the only value two runs cannot agree on is the CURRENT_TIMESTAMP default,
    which the hash masks and the previous test asserts non-empty.
    """
    source = tmp_path / "source.db"
    _seed_historical(source, 45, ["x", "y"])
    unseeded = tmp_path / "unseeded.db"
    seeded = tmp_path / "seeded.db"
    shutil.copy2(source, unseeded)
    shutil.copy2(source, seeded)

    CharactersRAGDB(unseeded, client_id="t21441-cmp").close_connection()
    CharactersRAGDB(
        seeded,
        client_id="t21441-cmp",
        console_library_migration_seed=ConsoleLibraryMigrationSeed(
            auto_retrieve_on_send=False
        ),
    ).close_connection()

    assert _content_hash(unseeded) == _content_hash(seeded)


def test_a_true_seed_still_carries_the_legacy_value(tmp_path: Path):
    """The seed is optional, not ignored: the boot path still wins."""
    db_path = tmp_path / "chachanotes.db"
    _seed_historical(db_path, 45, ["kept"])
    CharactersRAGDB(
        db_path,
        client_id="t21441-true",
        console_library_migration_seed=ConsoleLibraryMigrationSeed(
            auto_retrieve_on_send=True
        ),
    ).close_connection()

    assert {row[1] for row in _policy_rows(db_path)} == {1}


def test_a_wrong_typed_seed_is_still_rejected(tmp_path: Path):
    """An absent value defaults; a caller defect does not."""
    db_path = tmp_path / "chachanotes.db"
    _seed_historical(db_path, 45, ["bad"])
    with pytest.raises(SchemaError, match="ConsoleLibraryMigrationSeed"):
        CharactersRAGDB(
            db_path,
            client_id="t21441-bad",
            console_library_migration_seed={"auto_retrieve_on_send": True},
        )
    assert _raw_version(db_path) == 45
    assert "console_conversation_library_policy" not in _raw_table_names(db_path)


# ---------------------------------------------------------------------------
# migration correctness: atomic, re-enterable, interrupt-safe
# ---------------------------------------------------------------------------
def test_a_failure_inside_the_v48_step_rewinds_the_whole_chain(tmp_path: Path):
    """Deterministic atomicity: the chain is one transaction, not eight."""
    db_path = tmp_path / "chachanotes.db"
    _seed_historical(db_path, 45, ["rewind"])
    before = _content_hash(db_path)

    with patch.object(
        CharactersRAGDB,
        "_seed_console_library_policy_rows",
        side_effect=sqlite3.OperationalError("injected"),
    ):
        with pytest.raises(SchemaError):
            CharactersRAGDB(db_path, client_id="t21441-fail")

    assert _raw_version(db_path) == 45
    assert "console_conversation_library_policy" not in _raw_table_names(db_path)
    assert _content_hash(db_path) == before
    assert _integrity_is_clean(db_path)

    # ...and it is re-enterable: a plain retry still converges.
    CharactersRAGDB(db_path, client_id="t21441-retry").close_connection()
    assert _raw_version(db_path) == CURRENT


_STALL_CHILD = """
import sys, time
sys.path.insert(0, {repo!r})
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def stalled(self, cursor):
    print("inside-the-v48-transaction", flush=True)
    time.sleep(600)


CharactersRAGDB._update_console_library_policy_schema_version = stalled
CharactersRAGDB({path!r}, client_id="t21441-killed")
"""


def test_a_sigkill_inside_the_v48_transaction_cannot_brick_the_database(
    tmp_path: Path,
):
    """Real kill at the worst moment, not a simulated one.

    The child stalls in the last statement of the step -- the guarded version
    bump -- so the DDL has run and the policy rows are inserted when SIGKILL
    lands, with nothing committed. The parent proves that state through a
    read-only connection first (still v47, table not visible), because a
    partial apply that was ALREADY visible would be the brick this asserts
    against.
    """
    source = tmp_path / "source.db"
    _seed_historical(source, 47, ["k1", "k2", "k3"])
    killed = tmp_path / "killed.db"
    control = tmp_path / "control.db"
    shutil.copy2(source, killed)
    shutil.copy2(source, control)
    before = _content_hash(killed)

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _STALL_CHILD.format(repo=str(REPO_ROOT), path=str(killed)),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60.0
        marker = ""
        while time.monotonic() < deadline:
            marker = child.stdout.readline()
            if marker or child.poll() is not None:
                break
        assert "inside-the-v48-transaction" in marker, (
            f"child never entered the step (marker={marker!r} "
            f"stderr={child.stderr.read()!r})"
        )
        # The transaction is genuinely OPEN at kill time, not merely entered
        # and already rolled back: no second writer can take the lock. Without
        # this witness the assertions below would also pass against a child
        # that never started the step.
        with closing(sqlite3.connect(killed, timeout=0.2, isolation_level=None)) as other:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                other.execute("BEGIN IMMEDIATE")
        # Uncommitted work must be invisible to every other reader.
        assert _raw_version(killed) == 47
        assert "console_conversation_library_policy" not in _raw_table_names(killed)
        os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=60)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=60)

    # Nothing survived the kill, and the file is not damaged.
    assert _raw_version(killed) == 47
    assert "console_conversation_library_policy" not in _raw_table_names(killed)
    assert _integrity_is_clean(killed)
    assert _content_hash(killed) == before

    # Re-entering converges on exactly the uninterrupted result.
    CharactersRAGDB(killed, client_id="t21441-recovered").close_connection()
    CharactersRAGDB(control, client_id="t21441-control").close_connection()
    assert _raw_version(killed) == CURRENT
    assert _integrity_is_clean(killed)
    assert _content_hash(killed) == _content_hash(control)


def test_reopening_an_upgraded_database_changes_nothing(tmp_path: Path):
    """Re-enterability at the top: an at-version open is a no-op."""
    db_path = tmp_path / "chachanotes.db"
    _seed_historical(db_path, 45, ["stable"])
    CharactersRAGDB(db_path, client_id="t21441-first").close_connection()
    once = _content_hash(db_path)
    CharactersRAGDB(db_path, client_id="t21441-second").close_connection()
    assert _content_hash(db_path) == once


# ---------------------------------------------------------------------------
# AC #2 -- the shipped writer populates the schema it is opened against
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("version", [45, 47, CURRENT])
def test_the_shipped_writer_populates_the_schema_it_is_given(
    tmp_path: Path, version: int
):
    """`add_message` builds its INSERT from the table, not from the newest one."""
    db_path = tmp_path / "chachanotes.db"
    with chachanotes_db_at_version(db_path, version, client_id="t21441-writer") as db:
        conversation_id = db.add_conversation({"title": "w", "character_id": 1})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "historical body",
            }
        )
        row = db.execute_query(
            "SELECT content, role, version, deleted FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        assert (row["content"], row["role"], row["version"], row["deleted"]) == (
            "historical body",
            "user",
            1,
            0,
        )
        columns = db._messages_table_columns()
    # The column the fixture is meant to be missing is genuinely missing --
    # otherwise this test would pass without exercising anything.
    assert ("assistant_generation_state" in columns) == (version >= 48)


def test_the_writer_refuses_to_drop_a_value_the_schema_cannot_hold(tmp_path: Path):
    """The omission is lossless or it raises; it is never a silent drop.

    This is what keeps a schema-adaptive writer from masking an incompletely
    migrated database: every column newer than the base schema is nullable, so
    a genuine defect arrives as a non-None value with nowhere to go.
    """
    db_path = tmp_path / "chachanotes.db"
    with chachanotes_db_at_version(db_path, 47, client_id="t21441-loss") as db:
        conversation_id = db.add_conversation({"title": "loss", "character_id": 1})
        with pytest.raises(SchemaError, match="assistant_generation_state"):
            db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "body",
                    "assistant_generation_state": "complete",
                }
            )
        assert (
            db.execute_query(
                "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()["n"]
            == 0
        )


def test_the_writer_still_names_every_current_column(tmp_path: Path):
    """At the current version nothing is dropped -- the adaptive path is inert.

    Pinned as a statement, not a round trip, so a future column that the
    writer stops naming is visible here rather than only in whatever feature
    test happens to read it back.
    """
    db = CharactersRAGDB(tmp_path / "chachanotes.db", client_id="t21441-current")
    try:
        query, params = db._messages_insert_statement(
            (("id", "x"), ("assistant_generation_state", None), ("content", "c"))
        )
        assert "assistant_generation_state" in query
        assert params == ("x", None, "c")
        assert db._messages_table_columns() >= {
            "id",
            "conversation_id",
            "content",
            "role",
            "provider_continuation_json",
            "assistant_generation_state",
        }
    finally:
        db.close_connection()
