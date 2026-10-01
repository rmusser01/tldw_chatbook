"""Private plugin registry behavior, using isolated real SQLite databases."""

import sqlite3
from pathlib import Path

import pytest


def test_secondary_owner_cannot_execute(tmp_path):
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    first, second = PluginRuntimeOwner(tmp_path), PluginRuntimeOwner(tmp_path)
    try:
        assert first.try_acquire()
        assert not second.try_acquire()
    finally:
        first.close()
        second.close()


def test_memory_schema_pagination_and_rollback():
    from tldw_chatbook.Plugins.registry import PluginRegistry

    with PluginRegistry(Path(":memory:")) as registry:
        assert registry.schema_version == 1
        with registry.transaction() as cursor:
            cursor.executemany(
                "INSERT INTO installations(installation_id) VALUES (?)",
                [("b",), ("a",)],
            )
        assert registry.list_installations(limit=1, offset=0) == (
            {"installation_id": "a", "revision_digest": None, "activation_default": 0},
        )
        with pytest.raises(RuntimeError), registry.transaction() as cursor:
            cursor.execute("DELETE FROM installations")
            raise RuntimeError("rollback")
        assert len(registry.list_installations(limit=50, offset=0)) == 2
        for limit, offset in [(0, 0), (51, 0), (1, -1), (True, 0)]:
            with pytest.raises(ValueError):
                registry.list_installations(limit=limit, offset=offset)


def test_disk_requires_live_matching_owner_and_secondary_reads_wal(tmp_path):
    from tldw_chatbook.Plugins.registry import PluginRegistry
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    path = tmp_path / "registry.sqlite3"
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    with pytest.raises(PrivatePathError, match="missing_sqlite_artifact"):
        PluginRegistry(path)
    assert not path.exists()
    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        with PluginRegistry(path, owner=owner) as writer:
            with writer.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO installations(installation_id) VALUES ('test')"
                )
                assert cursor.execute("PRAGMA synchronous").fetchone()[0] == 2
                assert cursor.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            with PluginRegistry(path) as reader:
                assert (
                    reader.list_installations(limit=50, offset=0)[0]["installation_id"]
                    == "test"
                )
                with pytest.raises(PermissionError), reader.transaction():
                    pass
            assert path.stat().st_mode & 0o777 == 0o600
            assert Path(str(path) + "-wal").stat().st_mode & 0o777 == 0o600
            owner.close()
            with pytest.raises(PermissionError), writer.transaction():
                pass
    finally:
        owner.close()


def test_reopen_rejects_wrong_version_and_missing_schema(tmp_path):
    from tldw_chatbook.Plugins.registry import PluginRegistry
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        path = tmp_path / "registry.sqlite3"
        with (
            PluginRegistry(path, owner=owner) as registry,
            registry.transaction() as cursor,
        ):
            cursor.execute("PRAGMA user_version=42")
        with pytest.raises(sqlite3.DatabaseError):
            PluginRegistry(path, owner=owner)
        with sqlite3.connect(path) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 42
            connection.execute("PRAGMA user_version=1")
            connection.execute("DROP TABLE receipts")
        with pytest.raises(sqlite3.DatabaseError):
            PluginRegistry(path)
    finally:
        owner.close()


def test_immutable_inspection_roundtrip_and_foreign_keys(native_package):
    import json

    from tldw_chatbook.Plugins.inspection import inspect_package
    from tldw_chatbook.Plugins.registry import PluginRegistry

    inspection = inspect_package(native_package())
    with PluginRegistry(Path(":memory:")) as registry:
        with registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO installations(installation_id, revision_digest) VALUES (?, ?)",
                ("installed", inspection.effective_digest),
            )
            cursor.execute(
                "INSERT INTO revisions VALUES (?, ?, ?)",
                (
                    "installed",
                    inspection.effective_digest,
                    json.dumps(inspection.model_dump(mode="json")),
                ),
            )
            for component in inspection.inventory.values():
                cursor.execute(
                    "INSERT INTO components VALUES (?, ?, ?, ?)",
                    (
                        "installed",
                        inspection.effective_digest,
                        component.component_id,
                        json.dumps(component.model_dump(mode="json")),
                    ),
                )
        with pytest.raises(sqlite3.IntegrityError), registry.transaction() as cursor:
            cursor.execute("UPDATE revisions SET inspection_json='{}'")
        with pytest.raises(sqlite3.IntegrityError), registry.transaction() as cursor:
            cursor.execute("UPDATE components SET definition_json='{}'")
        with pytest.raises(sqlite3.IntegrityError), registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO selections VALUES ('missing', 'missing', 'missing', 1)"
            )
        with registry.transaction() as cursor:
            stored = cursor.execute("SELECT inspection_json FROM revisions").fetchone()[
                0
            ]
            assert json.loads(stored) == inspection.model_dump(mode="json")
        assert (
            registry.list_installations(limit=1, offset=0)[0]["activation_default"] == 0
        )


def test_owner_identity_loss_rolls_back_and_corruption_is_preserved(tmp_path):
    from tldw_chatbook.Plugins.registry import PluginRegistry
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    path = tmp_path / "registry.sqlite3"
    with PluginRegistry(path, owner=owner) as registry:
        with pytest.raises(PermissionError), registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO installations(installation_id) VALUES ('rollback')"
            )
            owner.close()
        assert registry.list_installations(limit=1, offset=0) == ()
    path.write_bytes(b"not a database")
    before = path.read_bytes()
    assert owner.try_acquire()
    try:
        with pytest.raises(sqlite3.DatabaseError):
            PluginRegistry(path, owner=owner)
        assert path.read_bytes() == before
        other = tmp_path / "other"
        other.mkdir()
        with pytest.raises(PermissionError):
            PluginRegistry(other / "registry.sqlite3", owner=owner)
        assert not (other / "registry.sqlite3").exists()
    finally:
        owner.close()


def test_failed_schema_creation_rolls_back_all_ddl(tmp_path, monkeypatch):
    import tldw_chatbook.Plugins.registry as module
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    statements = module._schema_statements()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                module, "_schema_statements", lambda: (*statements, "invalid sql")
            )
            with pytest.raises(sqlite3.DatabaseError):
                module.PluginRegistry(tmp_path / "registry.sqlite3", owner=owner)
        with sqlite3.connect(tmp_path / "registry.sqlite3") as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
            assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
        with module.PluginRegistry(
            tmp_path / "registry.sqlite3", owner=owner
        ) as registry:
            assert registry.schema_version == 1
    finally:
        owner.close()


def test_explicit_inherit_and_typed_authority_scopes_are_distinct():
    from tldw_chatbook.Plugins.registry import PluginRegistry

    with PluginRegistry(Path(":memory:")) as registry:
        with registry.transaction() as cursor:
            cursor.execute("INSERT INTO installations(installation_id) VALUES ('i')")
            cursor.executemany(
                "INSERT INTO activation VALUES (?, ?, ?)",
                [
                    ("i", "inherit", "inherit"),
                    ("i", "off", "disabled"),
                    ("i", "on", "enabled"),
                ],
            )
            assert (
                cursor.execute(
                    "SELECT intent FROM activation WHERE workspace_id='missing'"
                ).fetchone()
                is None
            )
            assert (
                cursor.execute(
                    "SELECT intent FROM activation WHERE workspace_id='inherit'"
                ).fetchone()[0]
                == "inherit"
            )
            assert (
                cursor.execute(
                    "SELECT intent FROM activation WHERE workspace_id='off'"
                ).fetchone()[0]
                == "disabled"
            )
            cursor.executemany(
                "INSERT INTO authority_generations(installation_id, scope_kind, workspace_id, generation) VALUES (?, ?, ?, ?)",
                [
                    ("i", "installation", "", 1),
                    ("i", "global_default", "", 2),
                    ("i", "workspace", "installation", 3),
                ],
            )
            assert (
                cursor.execute("SELECT count(*) FROM authority_generations").fetchone()[
                    0
                ]
                == 3
            )
        with pytest.raises(sqlite3.IntegrityError), registry.transaction() as cursor:
            cursor.execute("INSERT INTO activation VALUES ('i', 'bad', 'unknown')")
        with pytest.raises(sqlite3.IntegrityError), registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO authority_generations VALUES ('i', 'workspace', '', 0, 0)"
            )


@pytest.mark.parametrize(
    "escape",
    [
        "script",
        "COMMIT",
        "END",
        "ROLLBACK",
        "BEGIN",
        "SAVEPOINT early",
        "RELEASE early",
        "ROLLBACK TO early",
        "connection_commit",
        "connection_rollback",
    ],
)
def test_disk_transaction_blocks_early_boundary_escape(tmp_path, escape):
    from tldw_chatbook.Plugins.registry import PluginRegistry
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    path = tmp_path / "registry.sqlite3"
    try:
        with (
            PluginRegistry(path, owner=owner) as writer,
            PluginRegistry(path) as reader,
        ):
            # Positive control: both parameterized APIs stay invisible until exit.
            with writer.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO installations(installation_id) VALUES (?)",
                    ("control",),
                )
                cursor.executemany(
                    "INSERT INTO installations(installation_id) VALUES (?)",
                    [("control-2",), ("control-3",)],
                )
                assert reader.list_installations(limit=50, offset=0) == ()
            committed = reader.list_installations(limit=50, offset=0)
            assert len(committed) == 3
            with (
                pytest.raises(RuntimeError, match="body failed"),
                writer.transaction() as cursor,
            ):
                cursor.execute(
                    "INSERT INTO installations(installation_id) VALUES (?)",
                    ("must-rollback",),
                )
                with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                    if escape == "script":
                        cursor.executescript(
                            "INSERT INTO installations(installation_id) VALUES ('script-escape');"
                        )
                    elif escape == "connection_commit":
                        cursor.connection.commit()
                    elif escape == "connection_rollback":
                        cursor.connection.rollback()
                    else:
                        cursor.execute(escape)
                assert reader.list_installations(limit=50, offset=0) == committed
                raise RuntimeError("body failed")
            assert reader.list_installations(limit=50, offset=0) == committed
            # A denied statement must not leave the owner transaction unusable.
            with writer.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO installations(installation_id) VALUES (?)",
                    ("after-denial",),
                )
            assert len(reader.list_installations(limit=50, offset=0)) == 4
    finally:
        owner.close()


def test_disk_transaction_implicit_rollback_cannot_enable_autocommit(tmp_path):
    from tldw_chatbook.Plugins.registry import PluginRegistry
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    path = tmp_path / "registry.sqlite3"
    insert = "INSERT INTO installations(installation_id) VALUES (?)"
    try:
        with (
            PluginRegistry(path, owner=owner) as writer,
            PluginRegistry(path) as reader,
        ):
            with writer.transaction() as cursor:
                # Warm this exact statement to catch cached-authorization bypasses.
                cursor.execute(insert, ("existing",))
            with (
                pytest.raises(RuntimeError, match="body failed"),
                writer.transaction() as cursor,
            ):
                cursor.execute(insert, ("will-rollback",))
                with pytest.raises(sqlite3.IntegrityError):
                    cursor.execute(
                        "INSERT OR ROLLBACK INTO installations(installation_id) VALUES (?)",
                        ("existing",),
                    )
                with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
                    cursor.execute(insert, ("escaped",))
                raise RuntimeError("body failed")
            assert [
                row["installation_id"]
                for row in reader.list_installations(limit=50, offset=0)
            ] == ["existing"]
    finally:
        owner.close()


def test_caught_implicit_rollback_cannot_report_success(tmp_path):
    from tldw_chatbook.Plugins.registry import PluginRegistry
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        with PluginRegistry(tmp_path / "registry.sqlite3", owner=owner) as registry:
            with (
                pytest.raises(RuntimeError, match="transaction ended"),
                registry.transaction() as cursor,
            ):
                cursor.execute(
                    "INSERT INTO installations(installation_id) VALUES ('duplicate')"
                )
                with pytest.raises(sqlite3.IntegrityError):
                    cursor.execute(
                        "INSERT OR ROLLBACK INTO installations(installation_id) VALUES ('duplicate')"
                    )
            assert registry.list_installations(limit=50, offset=0) == ()
    finally:
        owner.close()
