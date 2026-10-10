"""Finite callbacks retire replacement handles, not unchanged native borrowers."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.UI.Console_Modules.character_context import (
    ConsoleCharacterContextController,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm-borrower"])
def test_metadata_retires_replacement_after_exact_file_quiescence(tmp_path, borrowed):
    """A stale borrowed flag must not retain a reopened worker connection."""
    db = CharactersRAGDB(tmp_path / "departed-profile.sqlite", "late-metadata")
    db.close_connection()
    entered, release = threading.Event(), threading.Event()

    def read():
        previous = db.get_connection() if borrowed else None
        with operation_owned_connection(db):
            entered.set()
            assert release.wait(5), "metadata callback was not released"
            metadata = ConsoleCharacterContextController._read_database_scope_metadata(
                db
            )
            replacement = db._local.conn
        return previous, replacement, metadata

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read)
        try:
            assert entered.wait(5), "callback did not enter its ownership guard"
            assert db.registered_connection_count() == int(borrowed)
            with db.quiesce_connections(timeout_seconds=5):
                assert db.registered_connection_count() == 0
            release.set()
            previous, replacement, metadata = future.result(timeout=5)
            assert metadata[0] and metadata[1] >= 0
            assert replacement is not previous
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                replacement.execute("SELECT 1")
            assert db.registered_connection_count() == 0
        finally:
            release.set()
            try:
                future.result(timeout=5)
            finally:
                # Settle physical work before retiring this exact test-owned file.
                with db.quiesce_connections(timeout_seconds=5):
                    pass


@pytest.mark.parametrize(
    "database_type", [AgentRunsDB, WorkspaceDB, LibraryCollectionsDB, MediaDatabase]
)
@pytest.mark.parametrize("replace", [False, True], ids=["unchanged", "replacement"])
@pytest.mark.parametrize("fail_sql", [False, True], ids=["success", "sql-error"])
def test_core_guard_retires_only_replacement_borrowers(
    tmp_path, database_type, replace, fail_sql
):
    """Cleanup follows native identity on success/error, preserving live transactions."""
    db = database_type(tmp_path / "core-owner.sqlite", "finite-core")
    get_connection = (
        db.get_connection if database_type is MediaDatabase else db._held_connection
    )
    try:
        previous = get_connection()
        previous.execute("BEGIN")
        error = (
            pytest.raises(sqlite3.OperationalError, match="no such table")
            if fail_sql
            else nullcontext()
        )
        with error, operation_owned_connection(db):
            if replace:
                db.close()
            current = get_connection()
            assert current.execute("SELECT 1").fetchone()[0] == 1
            if fail_sql:
                # Let an actual SQLite error cross the guard's finally boundary.
                current.execute("SELECT * FROM missing_finite_operation_table")
        if replace:
            assert current is not previous
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                current.execute("SELECT 1")
        else:
            assert current is previous
            assert current.in_transaction
            assert current.execute("SELECT 1").fetchone()[0] == 1
            current.rollback()
    finally:
        db.close()
