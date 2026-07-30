from __future__ import annotations

import os
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
from tldw_chatbook.Utils.private_paths import PrivatePathError, PrivatePathStatus
from tldw_chatbook.DB import (
    ChaChaNotes_DB,
    Client_Media_DB_v2,
    Evals_DB,
    Library_Ingest_Jobs_DB,
    Prompts_DB,
    RAG_Indexing_DB,
    base_db,
    search_history_db,
)


@dataclass(frozen=True)
class OwnerCase:
    name: str
    module: ModuleType
    owner_id: str
    expected_connect_kwargs: dict[str, Any]
    expected_pragmas: dict[str, Any]
    reuses_connection: bool


CORE_OWNER_CASES = (
    OwnerCase("base", base_db, "db.base", {}, {}, False),
    OwnerCase(
        "chachanotes",
        ChaChaNotes_DB,
        "db.chachanotes.primary",
        {
            "detect_types": sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            "check_same_thread": False,
            "timeout": 15,
        },
        {"foreign_keys": 1, "journal_mode": "wal"},
        True,
    ),
    OwnerCase(
        "media",
        Client_Media_DB_v2,
        "db.media.primary",
        {
            "detect_types": sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            "check_same_thread": False,
            "timeout": 10,
        },
        {"foreign_keys": 1, "journal_mode": "wal"},
        True,
    ),
    OwnerCase(
        "prompts",
        Prompts_DB,
        "db.prompts.primary",
        {
            "detect_types": sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            "check_same_thread": False,
            "timeout": 10,
        },
        {"foreign_keys": 1, "journal_mode": "wal"},
        True,
    ),
    OwnerCase(
        "evals",
        Evals_DB,
        "db.evals",
        {"check_same_thread": False},
        {"foreign_keys": 1, "journal_mode": "wal"},
        True,
    ),
    OwnerCase(
        "library_ingest_jobs",
        Library_Ingest_Jobs_DB,
        "db.library_ingest_jobs",
        {"check_same_thread": False},
        {"journal_mode": "wal", "synchronous": 1},
        True,
    ),
    OwnerCase("rag_indexing", RAG_Indexing_DB, "db.rag_indexing", {}, {}, False),
    OwnerCase("search_history", search_history_db, "db.search_history", {}, {}, False),
)


def _connection_for(case: OwnerCase, database: Any) -> sqlite3.Connection:
    if case.name in {"chachanotes", "media", "prompts"}:
        return database.get_connection()
    return (
        database.get_connection()
        if hasattr(database, "get_connection")
        else database._get_connection()
    )


def _close_database(case: OwnerCase, database: Any) -> None:
    if case.name in {"chachanotes", "media", "prompts"}:
        database.close_connection()
    elif case.name == "evals":
        connection = getattr(database._local, "connection", None)
        if connection is not None:
            connection.close()
            del database._local.connection
    elif case.name == "library_ingest_jobs":
        database.close()


def _patch_minimal_schema(
    case: OwnerCase,
    monkeypatch: pytest.MonkeyPatch,
    observed_modes: list[int],
) -> None:
    def initialize(database: Any) -> None:
        connection = _connection_for(case, database)
        is_memory = (
            database.db_path == ":memory:"
            if case.name == "evals"
            else database.is_memory_db
        )
        if not is_memory:
            selected_path = (
                database.db_path if case.name == "evals" else database.db_path_str
            )
            observed_modes.append(stat.S_IMODE(Path(selected_path).stat().st_mode))
        connection.execute("CREATE TABLE IF NOT EXISTS privacy_probe (value TEXT)")
        connection.commit()
        if not case.reuses_connection:
            connection.close()

    if case.name == "base":
        return
    schema_owner, method_name = {
        "chachanotes": (ChaChaNotes_DB.CharactersRAGDB, "_initialize_schema"),
        "media": (Client_Media_DB_v2.MediaDatabase, "_initialize_schema"),
        "prompts": (Prompts_DB.PromptsDatabase, "_initialize_schema"),
        "evals": (Evals_DB.EvalsDB, "_init_schema"),
        "library_ingest_jobs": (
            Library_Ingest_Jobs_DB.LibraryIngestJobsDB,
            "_initialize_schema",
        ),
        "rag_indexing": (RAG_Indexing_DB.RAGIndexingDB, "_initialize_schema"),
        "search_history": (
            search_history_db.SearchHistoryDB,
            "_initialize_schema",
        ),
    }[case.name]
    monkeypatch.setattr(schema_owner, method_name, initialize)


def _construct(
    case: OwnerCase,
    target: str | Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_modes: list[int],
) -> Any:
    _patch_minimal_schema(case, monkeypatch, observed_modes)
    if case.name == "base":

        class ConcreteBaseDB(base_db.BaseDB):
            def _initialize_schema(self) -> None:
                connection = self._get_connection()
                if not self.is_memory_db:
                    observed_modes.append(
                        stat.S_IMODE(Path(self.db_path_str).stat().st_mode)
                    )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS privacy_probe (value TEXT)"
                )
                connection.commit()
                connection.close()

        return ConcreteBaseDB(target)
    if case.name == "chachanotes":
        return ChaChaNotes_DB.CharactersRAGDB(target, "privacy-test")
    if case.name == "media":
        return Client_Media_DB_v2.MediaDatabase(target, "privacy-test")
    if case.name == "prompts":
        return Prompts_DB.PromptsDatabase(target, "privacy-test")
    if case.name == "evals":
        return Evals_DB.EvalsDB(target, "privacy-test")
    if case.name == "library_ingest_jobs":
        return Library_Ingest_Jobs_DB.LibraryIngestJobsDB(target, "privacy-test")
    if case.name == "rag_indexing":
        return RAG_Indexing_DB.RAGIndexingDB(target, "privacy-test")
    if case.name == "search_history":
        return search_history_db.SearchHistoryDB(target, "privacy-test")
    raise AssertionError(case.name)


@pytest.mark.skipif(os.name != "posix", reason="POSIX private file contract")
@pytest.mark.parametrize("existing", [False, True], ids=["first-create", "harden"])
@pytest.mark.parametrize("case", CORE_OWNER_CASES, ids=lambda case: case.name)
def test_core_owner_uses_literal_seam_and_preserves_connection_contract(
    case: OwnerCase,
    existing: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / case.name
    parent.mkdir(mode=0o750)
    parent_mode = stat.S_IMODE(parent.stat().st_mode)
    target = parent / "owner.sqlite"
    if existing:
        target.write_bytes(b"")
        target.chmod(0o644)

    seen: list[tuple[str, str | os.PathLike[str], dict[str, Any]]] = []
    real_connect = private_sqlite.connect_private_sqlite

    def recording_connect(
        owner_id: str,
        database: str | os.PathLike[str],
        **kwargs: Any,
    ) -> sqlite3.Connection:
        seen.append((owner_id, database, kwargs.copy()))
        return real_connect(owner_id, database, **kwargs)

    monkeypatch.setattr(case.module, "connect_private_sqlite", recording_connect)
    observed_modes: list[int] = []
    previous_umask = os.umask(0)
    try:
        database = _construct(case, target, monkeypatch, observed_modes)
    finally:
        os.umask(previous_umask)

    try:
        connection = _connection_for(case, database)
        assert isinstance(connection.execute("SELECT 1").fetchone(), sqlite3.Row)
        for pragma, expected in case.expected_pragmas.items():
            actual = connection.execute(f"PRAGMA {pragma}").fetchone()[0]
            assert str(actual).lower() == str(expected).lower()
        if case.reuses_connection:
            assert _connection_for(case, database) is connection
        else:
            connection.close()

        assert seen
        assert {owner_id for owner_id, _path, _kwargs in seen} == {case.owner_id}
        assert all(
            os.fspath(path) == os.fspath(target) for _owner_id, path, _kwargs in seen
        )
        assert all(kwargs == case.expected_connect_kwargs for _, _, kwargs in seen)
        assert observed_modes == [0o600]
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert stat.S_IMODE(parent.stat().st_mode) == parent_mode
        assert database.db_path == target
    finally:
        _close_database(case, database)


@pytest.mark.parametrize("case", CORE_OWNER_CASES, ids=lambda case: case.name)
def test_core_owner_preserves_exact_memory_behavior(
    case: OwnerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str | os.PathLike[str], dict[str, Any]]] = []
    real_connect = private_sqlite.connect_private_sqlite

    def recording_connect(
        owner_id: str,
        database: str | os.PathLike[str],
        **kwargs: Any,
    ) -> sqlite3.Connection:
        seen.append((owner_id, database, kwargs.copy()))
        return real_connect(owner_id, database, **kwargs)

    monkeypatch.setattr(case.module, "connect_private_sqlite", recording_connect)
    database = _construct(case, ":memory:", monkeypatch, [])
    try:
        connection = _connection_for(case, database)
        if not case.reuses_connection:
            connection.execute("CREATE TABLE IF NOT EXISTS privacy_probe (value TEXT)")
        connection.execute("INSERT INTO privacy_probe VALUES ('memory')")
        assert (
            connection.execute("SELECT value FROM privacy_probe").fetchone()[0]
            == "memory"
        )
        assert seen
        assert {owner_id for owner_id, _path, _kwargs in seen} == {case.owner_id}
        assert all(path == ":memory:" for _owner_id, path, _kwargs in seen)
    finally:
        if not case.reuses_connection:
            connection.close()
        _close_database(case, database)


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace contract")
@pytest.mark.parametrize(
    "unsafe_kind",
    ["missing_parent", "writable_parent", "symlink_parent", "symlink_target"],
)
@pytest.mark.parametrize("case", CORE_OWNER_CASES, ids=lambda case: case.name)
def test_core_owner_rejects_unsafe_namespace_before_raw_sqlite(
    case: OwnerCase,
    unsafe_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / f"{case.name}-outside"
    outside.mkdir()
    outside_target = outside / "outside.sqlite"
    outside_target.write_bytes(b"outside")
    if unsafe_kind == "missing_parent":
        parent = tmp_path / case.name / "missing"
        target = parent / "owner.sqlite"
    elif unsafe_kind == "writable_parent":
        parent = tmp_path / case.name
        parent.mkdir()
        parent.chmod(0o777)
        target = parent / "owner.sqlite"
    elif unsafe_kind == "symlink_parent":
        parent = tmp_path / case.name
        parent.symlink_to(outside, target_is_directory=True)
        target = parent / "owner.sqlite"
    else:
        parent = tmp_path / case.name
        parent.mkdir()
        target = parent / "owner.sqlite"
        target.symlink_to(outside_target)

    raw_connect_calls = 0

    def forbidden_raw_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal raw_connect_calls
        raw_connect_calls += 1
        raise sqlite3.OperationalError("raw SQLite reached unsafe namespace")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", forbidden_raw_connect)
    expected_exception = {
        "chachanotes": ChaChaNotes_DB.CharactersRAGDBError,
        "media": Client_Media_DB_v2.DatabaseError,
        "prompts": Prompts_DB.DatabaseError,
    }.get(case.name, PrivatePathError)
    with pytest.raises(expected_exception) as caught:
        _construct(case, target, monkeypatch, [])

    boundary_error: BaseException = caught.value
    while boundary_error.__cause__ is not None:
        boundary_error = boundary_error.__cause__
    assert isinstance(boundary_error, PrivatePathError)
    expected_status = (
        PrivatePathStatus.UNSAFE_PARENT
        if unsafe_kind in {"missing_parent", "writable_parent"}
        else PrivatePathStatus.LINK_OR_NON_REGULAR
    )
    assert boundary_error.result.status is expected_status
    assert raw_connect_calls == 0
    if unsafe_kind == "missing_parent":
        assert not parent.exists()
    assert outside_target.read_bytes() == b"outside"


@pytest.mark.parametrize("case", CORE_OWNER_CASES, ids=lambda case: case.name)
def test_core_owner_never_resolves_selected_path(
    case: OwnerCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / case.name / "owner.sqlite"
    target.parent.mkdir()
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda *args, **kwargs: pytest.fail("core owner called Path.resolve()"),
    )
    database = _construct(case, target, monkeypatch, [])
    _close_database(case, database)


@pytest.mark.parametrize("case", CORE_OWNER_CASES, ids=lambda case: case.name)
def test_core_owner_normalizes_relative_path_lexically(
    case: OwnerCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_target = Path(case.name) / "owner.sqlite"
    relative_target.parent.mkdir()
    expected_target = tmp_path / relative_target
    seen: list[str] = []
    prepared: list[Path] = []
    raw_seen: list[str] = []
    real_connect = private_sqlite.connect_private_sqlite
    real_prepare = private_sqlite._prepare_artifact
    real_raw_connect = sqlite3.connect

    def recording_connect(
        owner_id: str,
        database: str | os.PathLike[str],
        **kwargs: Any,
    ) -> sqlite3.Connection:
        del owner_id
        seen.append(os.fspath(database))
        return real_connect(case.owner_id, database, **kwargs)

    def recording_prepare(selected: Path, **kwargs: Any) -> bool:
        prepared.append(selected)
        return real_prepare(selected, **kwargs)

    def recording_raw_connect(
        database: str | os.PathLike[str],
        *args: Any,
        **kwargs: Any,
    ) -> sqlite3.Connection:
        raw_seen.append(os.fspath(database))
        return real_raw_connect(database, *args, **kwargs)

    monkeypatch.setattr(case.module, "connect_private_sqlite", recording_connect)
    monkeypatch.setattr(private_sqlite, "_prepare_artifact", recording_prepare)
    monkeypatch.setattr(private_sqlite.sqlite3, "connect", recording_raw_connect)
    database = _construct(case, relative_target, monkeypatch, [])
    try:
        expected_stored_path = (
            relative_target if case.name == "evals" else expected_target
        )
        expected_owner_target = (
            relative_target if case.name == "evals" else expected_target
        )
        assert database.db_path == expected_stored_path
        assert seen
        assert set(seen) == {os.fspath(expected_owner_target)}
        assert expected_target in prepared
        assert all(path.is_absolute() for path in prepared)
        assert raw_seen
        assert set(raw_seen) == {os.fspath(expected_target)}
    finally:
        _close_database(case, database)


@pytest.mark.parametrize(
    ("case", "expected_exception"),
    [
        (CORE_OWNER_CASES[0], sqlite3.OperationalError),
        (CORE_OWNER_CASES[1], ChaChaNotes_DB.CharactersRAGDBError),
        (CORE_OWNER_CASES[2], Client_Media_DB_v2.DatabaseError),
        (CORE_OWNER_CASES[3], Prompts_DB.DatabaseError),
        (CORE_OWNER_CASES[4], sqlite3.OperationalError),
        (CORE_OWNER_CASES[5], sqlite3.OperationalError),
        (CORE_OWNER_CASES[6], sqlite3.OperationalError),
        (CORE_OWNER_CASES[7], sqlite3.OperationalError),
    ],
    ids=lambda value: value.name if isinstance(value, OwnerCase) else value.__name__,
)
def test_domain_connection_exception_translation_is_preserved(
    case: OwnerCase,
    expected_exception: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / f"{case.name}.sqlite"
    monkeypatch.setattr(
        case.module,
        "connect_private_sqlite",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            sqlite3.OperationalError("connection failed")
        ),
    )

    with pytest.raises(expected_exception) as caught:
        _construct(case, target, monkeypatch, [])

    root_cause: BaseException = caught.value
    while root_cause.__cause__ is not None:
        root_cause = root_cause.__cause__
    assert isinstance(root_cause, sqlite3.OperationalError)


@pytest.mark.skipif(os.name != "posix", reason="POSIX reconnect boundary contract")
@pytest.mark.parametrize(
    ("case", "expected_exception"),
    [
        (CORE_OWNER_CASES[1], ChaChaNotes_DB.CharactersRAGDBError),
        (CORE_OWNER_CASES[2], Client_Media_DB_v2.DatabaseError),
        (CORE_OWNER_CASES[3], Prompts_DB.DatabaseError),
    ],
    ids=lambda value: value.name if isinstance(value, OwnerCase) else value.__name__,
)
def test_domain_connection_exception_translation_is_preserved_on_reconnect(
    case: OwnerCase,
    expected_exception: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / case.name
    parent.mkdir(mode=0o750)
    target = parent / "owner.sqlite"
    database = _construct(case, target, monkeypatch, [])
    database.close_connection()
    parent.chmod(0o777)
    raw_connect_calls = 0

    def forbidden_raw_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal raw_connect_calls
        raw_connect_calls += 1
        raise sqlite3.OperationalError("raw SQLite reached unsafe reconnect")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", forbidden_raw_connect)
    try:
        with pytest.raises(expected_exception) as caught:
            database.get_connection()

        root_cause: BaseException = caught.value
        while root_cause.__cause__ is not None:
            root_cause = root_cause.__cause__
        assert isinstance(root_cause, PrivatePathError)
        assert root_cause.result.status is PrivatePathStatus.UNSAFE_PARENT
        assert raw_connect_calls == 0
    finally:
        parent.chmod(0o750)
        database.close_connection()


@dataclass(frozen=True)
class BackupCase:
    name: str
    module: ModuleType
    owner: OwnerCase
    backup_owner_id: str


BACKUP_CASES = (
    BackupCase(
        "chachanotes",
        ChaChaNotes_DB,
        CORE_OWNER_CASES[1],
        "db.chachanotes.backup",
    ),
    BackupCase(
        "media",
        Client_Media_DB_v2,
        CORE_OWNER_CASES[2],
        "db.media.backup",
    ),
    BackupCase(
        "prompts",
        Prompts_DB,
        CORE_OWNER_CASES[3],
        "db.prompts.backup",
    ),
)


def test_chachanotes_backup_reopen_preserves_conversation_authority(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir(mode=0o700)
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(mode=0o700)
    source_path = source_dir / "chachanotes.sqlite"
    backup_path = backup_dir / "chachanotes.sqlite"
    authority_id = "server-user-v1:" + ("d" * 64)

    source_db = ChaChaNotes_DB.CharactersRAGDB(source_path, "source-client")
    try:
        conversation_id = source_db.add_conversation(
            {
                "assistant_kind": "character",
                "assistant_id": "opaque-server-character",
                "assistant_authority_id": authority_id,
                "runtime_backend": "server",
                "title": "Backed up character chat",
            }
        )
        assert source_db.backup_database(str(backup_path)) is True
    finally:
        source_db.close_connection()

    restored_db = ChaChaNotes_DB.CharactersRAGDB(backup_path, "restored-client")
    try:
        columns = {
            row["name"]
            for row in restored_db.get_connection()
            .execute("PRAGMA table_info(conversations)")
            .fetchall()
        }
        restored = restored_db.get_conversation_by_id(conversation_id)

        assert "assistant_authority_id" in columns
        assert restored["assistant_authority_id"] == authority_id
    finally:
        restored_db.close_connection()


@pytest.mark.skipif(os.name != "posix", reason="POSIX backup target contract")
@pytest.mark.parametrize("existing", [False, True], ids=["first-create", "harden"])
@pytest.mark.parametrize("backup_case", BACKUP_CASES, ids=lambda case: case.name)
def test_backup_target_uses_centralized_private_backup_helper(
    backup_case: BackupCase,
    existing: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / f"{backup_case.name}-source.sqlite"
    database = _construct(backup_case.owner, source, monkeypatch, [])
    source_connection = database.get_connection()
    source_connection.execute("INSERT INTO privacy_probe VALUES ('copied')")
    source_connection.commit()
    target_parent = tmp_path / f"{backup_case.name}-backups"
    target_parent.mkdir(mode=0o750)
    parent_mode = stat.S_IMODE(target_parent.stat().st_mode)
    target = target_parent / "backup.sqlite"
    if existing:
        target.write_bytes(b"")
        target.chmod(0o644)

    seen: list[tuple[str, sqlite3.Connection, Path, Path]] = []
    real_backup = private_sqlite.backup_connection_to_private

    def recording_backup(
        owner_id: str,
        connection: sqlite3.Connection,
        source_database: str | os.PathLike[str],
        target_database: str | os.PathLike[str],
    ) -> None:
        seen.append(
            (
                owner_id,
                connection,
                Path(source_database),
                Path(target_database),
            )
        )
        real_backup(
            owner_id,
            connection,
            source_database,
            target_database,
        )

    monkeypatch.setattr(
        backup_case.module,
        "backup_connection_to_private",
        recording_backup,
    )
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda *args, **kwargs: pytest.fail("backup path called Path.resolve()"),
    )
    previous_umask = os.umask(0)
    try:
        assert database.backup_database(str(target)) is True
    finally:
        os.umask(previous_umask)
        _close_database(backup_case.owner, database)

    assert seen == [
        (
            backup_case.backup_owner_id,
            source_connection,
            source,
            target,
        )
    ]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target_parent.stat().st_mode) == parent_mode
    with sqlite3.connect(target) as copied:
        assert copied.execute("SELECT value FROM privacy_probe").fetchone() == (
            "copied",
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX backup namespace contract")
@pytest.mark.parametrize(
    "unsafe_kind", ["missing_parent", "writable_parent", "symlink"]
)
@pytest.mark.parametrize("backup_case", BACKUP_CASES, ids=lambda case: case.name)
def test_backup_target_rejects_unsafe_namespace_before_raw_sqlite(
    backup_case: BackupCase,
    unsafe_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / f"{backup_case.name}-source.sqlite"
    database = _construct(backup_case.owner, source, monkeypatch, [])
    if unsafe_kind == "missing_parent":
        parent = tmp_path / "missing"
        target = parent / "backup.sqlite"
    elif unsafe_kind == "writable_parent":
        parent = tmp_path / "writable"
        parent.mkdir()
        parent.chmod(0o777)
        target = parent / "backup.sqlite"
    else:
        parent = tmp_path / "backups"
        parent.mkdir()
        outside = tmp_path / "outside.sqlite"
        outside.write_bytes(b"outside")
        target = parent / "backup.sqlite"
        target.symlink_to(outside)

    raw_connect_calls = 0

    def forbidden_raw_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        nonlocal raw_connect_calls
        raw_connect_calls += 1
        raise sqlite3.OperationalError("raw SQLite reached unsafe backup")

    monkeypatch.setattr(private_sqlite.sqlite3, "connect", forbidden_raw_connect)
    try:
        assert database.backup_database(str(target)) is False
        assert raw_connect_calls == 0
        if unsafe_kind == "missing_parent":
            assert not parent.exists()
    finally:
        _close_database(backup_case.owner, database)
