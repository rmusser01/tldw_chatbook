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
    """Catch an opener that could bypass the config-layer seed sanitization.

    Still load-bearing after task-21441 made the seed optional, and arguably
    more so: an absent seed now DEFAULTS (to automatic retrieval off) instead
    of raising, so a production opener that stopped passing one would silently
    discard a user's `chat_defaults.rag_auto_retrieve_on_send = true` on
    upgrade instead of failing loudly. This static sweep is what catches that.
    """

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


def _v47_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    conversation_title: str | None = None,
) -> tuple[Path, str | None]:
    """Build a genuine v47 database, optionally holding one conversation."""

    path = tmp_path / "v47.sqlite"
    conversation_id: str | None = None
    with monkeypatch.context() as v47_patch:
        v47_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 47)
        seeded = CharactersRAGDB(path, client_id="v47-seed")
        if conversation_title is not None:
            conversation_id = seeded.add_conversation(
                {"title": conversation_title, "character_id": 1}
            )
        seeded.close_connection()
    assert _db_version(path) == 47
    return path, conversation_id


def test_v47_upgrade_rejects_a_wrong_typed_seed_before_ddl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller defect still stops the migration before it touches the DDL."""

    path, _ = _v47_database(tmp_path, monkeypatch)
    before = _schema_snapshot(path)

    with monkeypatch.context() as target_patch:
        target_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 48)
        with pytest.raises(
            SchemaError,
            match="must be a ConsoleLibraryMigrationSeed",
        ):
            CharactersRAGDB(
                path,
                client_id="v48-upgrade",
                console_library_migration_seed=object(),  # type: ignore[arg-type]
            )

    assert _db_version(path) == 47
    assert _schema_snapshot(path) == before


def test_v47_upgrade_without_a_seed_migrates_with_retrieval_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The inverted half of this test, and the reason task-21441 exists.

    This case used to raise, which is what stopped a ChaChaNotes database from
    being upgradable by anything but the TUI. An absent seed now defaults to
    the config layer's own default -- automatic retrieval off -- so the
    migration completes and the policy it writes is the fail-safe one.
    """

    path, conversation_id = _v47_database(
        tmp_path, monkeypatch, conversation_title="legacy"
    )

    with monkeypatch.context() as target_patch:
        target_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 48)
        upgraded = CharactersRAGDB(path, client_id="v48-upgrade")
        upgraded.close_connection()

    assert _db_version(path) == 48
    with sqlite3.connect(path) as connection:
        policy = connection.execute(
            "SELECT auto_retrieve_on_send, assistant_library_access"
            " FROM console_conversation_library_policy WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    assert policy == (0, 1)
