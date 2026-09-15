"""Real workflow authoring persistence and foreign-writer isolation."""

import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Workflows.helpers import prompt_definition
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Workflows.document_service import DocumentService


def foreign_begin_immediate(path: Path) -> str:
    """Observe writer exclusion from an isolated, bounded stdlib-only child."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
try:
    try:
        connection.execute('BEGIN IMMEDIATE')
    except sqlite3.OperationalError as error:
        if error.sqlite_errorcode != sqlite3.SQLITE_BUSY:
            raise
        print('blocked')
    else:
        print('acquired')
        connection.rollback()
finally:
    connection.close()
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    assert result.stderr == ""
    assert result.stdout.strip() in {"blocked", "acquired"}
    return result.stdout.strip()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-owned SQLite locks")
@pytest.mark.parametrize("journal_mode", ["DELETE", "WAL"])
@pytest.mark.parametrize("operation", ["failed_constructor", "close_sibling"])
def test_ordinary_authoring_preserves_foreign_writer_exclusion(
    tmp_path, monkeypatch, journal_mode, operation
):
    module = importlib.import_module("tldw_chatbook.DB.Workflows_DB")
    path = tmp_path / "writer.sqlite3"
    setup = sqlite3.connect(path)
    try:
        assert setup.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[0] == (
            journal_mode.lower()
        )
    finally:
        setup.close()
    first = WorkflowsDB(path)
    second = WorkflowsDB(path)
    try:
        original = module.connect_private_sqlite

        def no_sqlite_wait(*args, **kwargs):
            return original(*args, timeout=0, **kwargs)

        monkeypatch.setattr(module, "connect_private_sqlite", no_sqlite_wait)
        with first.transaction() as cursor:
            assert cursor.execute("SELECT 1").fetchone()[0] == 1
            assert foreign_begin_immediate(path) == "blocked"
            if operation == "failed_constructor":
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    WorkflowsDB(path)
            else:
                second.close()
            assert foreign_begin_immediate(path) == "blocked"
            assert cursor.execute("SELECT 1").fetchone()[0] == 1
        assert foreign_begin_immediate(path) == "acquired"
    finally:
        second.close()
        first.close()


def test_restarting_store_keeps_saved_revision_and_exact_unfinished_draft(tmp_path):
    path = tmp_path / "authoring.sqlite3"
    db = WorkflowsDB(path)
    documents = DocumentService(db)
    try:
        revision = documents.create(json.dumps(prompt_definition()))
        documents.put_draft(
            revision.workflow_id, revision.revision_id, '{"unfinished":', 1
        )
    finally:
        db.close()
    reopened = WorkflowsDB(path)
    try:
        documents = DocumentService(reopened)
        assert (
            documents.get_revision(revision.workflow_id, revision.revision_id)
            == revision
        )
        draft = documents.get_draft(revision.workflow_id, revision.revision_id)
        assert draft.raw_text == '{"unfinished":'
        assert draft.last_valid_json == revision.raw_json
        assert draft.error is not None
    finally:
        reopened.close()
    assert not list(tmp_path.glob("*.lock"))


def test_migration_bytes_remain_compatible_with_existing_stores():
    migrations = Path(__file__).parents[2] / "tldw_chatbook/DB/migrations"
    expected = (
        "48203249df2308308823ce592db7561cb0563111975000c3a963d4360c2df832",
        "006fce4c8ac90ff5a629a604983be3716dbfd25989df3f32e874e215e2d85cc9",
        "c1c23ec7af2212583d7506818c5b8dc2980cc6e87dc5f304cb12f0238f731118",
        "9c4a4edb77a3e48b8b480e48e35291b7869790511337ccdb7fda3380fc3029ac",
    )
    for version, digest in enumerate(expected):
        path = migrations / f"workflows_v{version}_to_v{version + 1}.sql"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
