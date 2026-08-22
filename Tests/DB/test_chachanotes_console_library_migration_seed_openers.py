"""Regression coverage for explicit Console Library migration seed plumbing."""

import ast
import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, SchemaError


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_PRODUCTION_PACKAGE = _REPOSITORY_ROOT / "tldw_chatbook"
_SCHEMA_NAME = "rag_char_chat_schema"


def _production_characters_rag_db_calls() -> list[tuple[Path, ast.Call]]:
    """Return every direct production ``CharactersRAGDB`` constructor call."""

    calls: list[tuple[Path, ast.Call]] = []
    for source_path in _PRODUCTION_PACKAGE.rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "CharactersRAGDB":
                    calls.append((source_path, node))
    return calls


def _schema_snapshot(path: Path) -> tuple[int, tuple[tuple[str, str | None], ...]]:
    """Return the SQLite DDL state used to prove the guard ran before mutation."""

    with sqlite3.connect(path) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
        objects = tuple(
            connection.execute(
                "SELECT name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
        )
    return schema_version, objects


def _db_version(path: Path) -> int:
    """Read the persisted ChaChaNotes schema version."""

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (_SCHEMA_NAME,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_every_production_characters_rag_db_opener_passes_explicit_seed() -> None:
    """Catch an opener that could bypass the config-layer seed sanitization."""

    calls = _production_characters_rag_db_calls()
    assert calls
    missing_seed = [
        f"{source_path.relative_to(_REPOSITORY_ROOT)}:{call.lineno}"
        for source_path, call in calls
        if "console_library_migration_seed" not in {keyword.arg for keyword in call.keywords}
    ]

    assert missing_seed == []


def test_fresh_database_accepts_no_console_library_migration_seed(tmp_path: Path) -> None:
    """A new database has no legacy policy to migrate, so no seed is needed."""

    path = tmp_path / "fresh.sqlite"
    db = CharactersRAGDB(path, client_id="fresh")

    assert _db_version(path) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    db.close_connection()


def test_current_database_accepts_no_console_library_migration_seed(tmp_path: Path) -> None:
    """An already-current database does not need a seed on subsequent opens."""

    path = tmp_path / "current.sqlite"
    created = CharactersRAGDB(path, client_id="create")
    created.close_connection()

    reopened = CharactersRAGDB(path, client_id="reopen")

    assert _db_version(path) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    reopened.close_connection()


@pytest.mark.parametrize("migration_seed", [None, object()])
def test_v44_upgrade_rejects_missing_or_invalid_seed_before_ddl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    migration_seed: object,
) -> None:
    """A legacy v44 database must not start the v45 migration without a typed seed."""

    path = tmp_path / "v44.sqlite"
    seeded = CharactersRAGDB(path, client_id="v44-seed")
    seeded.close_connection()
    assert _db_version(path) == 44
    before = _schema_snapshot(path)

    with monkeypatch.context() as target_patch:
        target_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 45)
        with pytest.raises(
            SchemaError,
            match="Console library migration seed is required for v44 upgrade",
        ):
            CharactersRAGDB(
                path,
                client_id="v45-upgrade",
                console_library_migration_seed=migration_seed,  # type: ignore[arg-type]
            )

    assert _db_version(path) == 44
    assert _schema_snapshot(path) == before
