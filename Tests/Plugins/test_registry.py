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
        assert registry.schema_version == 2
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
            assert registry.schema_version == 2
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


def test_v1_upgrade_retains_rows_and_adds_independent_trust_and_tombstones(tmp_path):
    from tldw_chatbook.Plugins.registry import PluginRegistry, _schema_statements
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    path = tmp_path / "registry.sqlite3"
    with sqlite3.connect(path) as connection:
        for statement in _schema_statements():
            connection.execute(statement)
        connection.execute("PRAGMA user_version=1")
        connection.execute("INSERT INTO installations VALUES ('installed', NULL, 0)")
    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        with PluginRegistry(path, owner=owner) as registry:
            assert registry.schema_version == 2
            with registry.transaction() as cursor:
                assert (
                    cursor.execute("SELECT COUNT(*) FROM revision_trust").fetchone()[0]
                    == 0
                )
                cursor.execute(
                    "INSERT INTO tombstones VALUES ('removed', 3, 'remove-op')"
                )
            snapshot = registry.authority_projection(operation_result=None)
            assert snapshot["installations"] == [
                {
                    "installation_id": "installed",
                    "revision_digest": None,
                    "activation_default": False,
                }
            ]
            assert snapshot["tombstones"] == [
                {
                    "installation_id": "removed",
                    "generation": 3,
                    "operation_id": "remove-op",
                }
            ]
        with PluginRegistry(path) as registry:
            assert registry.schema_version == 2
            assert registry.authority_projection(operation_result=None) == snapshot
    finally:
        owner.close()


def test_projection_preserves_review_blockers_and_excludes_liveness(native_package):
    import json

    from tldw_chatbook.Plugins.inspection import inspect_package
    from tldw_chatbook.Plugins.registry import PluginRegistry

    inspection = inspect_package(
        native_package(requires={"skill:hello": ["hook:missing"]})
    )
    digest = inspection.effective_digest
    with PluginRegistry(Path(":memory:")) as registry:
        with registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO installations VALUES (?, ?, 0)", ("installed", digest)
            )
            cursor.execute(
                "INSERT INTO revisions VALUES (?, ?, ?)",
                ("installed", digest, inspection.model_dump_json()),
            )
            for component in inspection.inventory.values():
                cursor.execute(
                    "INSERT INTO components VALUES (?, ?, ?, ?)",
                    (
                        "installed",
                        digest,
                        component.component_id,
                        component.model_dump_json(),
                    ),
                )
            cursor.execute(
                "INSERT INTO revision_trust VALUES (?, ?, 0)", ("installed", digest)
            )
            cursor.execute(
                "INSERT INTO activation VALUES ('installed', 'work', 'disabled')"
            )
        before = registry.authority_projection(operation_result=None)
        assert before["revision_trust"][0]["reviewed"] is False
        assert before["activation"][0]["intent"] == "disabled"
        assert before["revisions"][0]["activation_blockers"] == list(
            inspection.activation_blockers
        )
        assert "evidence" not in json.dumps(before)
        assert all(
            "definition_digest" in c and "definition_json" not in c
            for c in before["components"]
        )
        with registry.transaction() as cursor:
            cursor.execute("UPDATE revision_trust SET reviewed=1")
        assert (
            registry.authority_projection(operation_result=None)["revision_trust"][0][
                "reviewed"
            ]
            is True
        )


def test_v1_migration_failure_is_transactional(tmp_path, monkeypatch):
    import tldw_chatbook.Plugins.registry as module
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    path = tmp_path / "registry.sqlite3"
    with sqlite3.connect(path) as connection:
        for statement in module._schema_statements():
            connection.execute(statement)
        connection.execute("PRAGMA user_version=1")
    original = module._schema_statements
    monkeypatch.setattr(
        module,
        "_schema_statements",
        lambda migration="001_initial.sql": (
            original(migration)
            + (("INVALID SQL",) if migration == "002_authority.sql" else ())
        ),
    )
    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        with pytest.raises(sqlite3.DatabaseError):
            module.PluginRegistry(path, owner=owner)
        with sqlite3.connect(path) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
            assert not connection.execute(
                "SELECT name FROM sqlite_master WHERE name IN ('revision_trust', 'tombstones')"
            ).fetchall()
    finally:
        owner.close()


def test_invalid_scalar_component_is_bound_as_blocked_reference(native_package):
    import json

    from tldw_chatbook.Plugins.inspection import inspect_package
    from tldw_chatbook.Plugins.registry import PluginRegistry

    root = native_package()
    (root / "mcp.json").write_text(
        json.dumps(
            {
                "$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                "mcpServers": {"bad": 42},
            }
        )
    )
    inspection = inspect_package(root)
    assert inspection.inventory["mcp:bad"].support == "invalid"
    with PluginRegistry(Path(":memory:")) as registry:
        with registry.transaction() as cursor:
            cursor.execute(
                "INSERT INTO installations VALUES ('installed', ?, 0)",
                (inspection.effective_digest,),
            )
            cursor.execute(
                "INSERT INTO revisions VALUES ('installed', ?, ?)",
                (inspection.effective_digest, inspection.model_dump_json()),
            )
            for component in inspection.inventory.values():
                cursor.execute(
                    "INSERT INTO components VALUES ('installed', ?, ?, ?)",
                    (
                        inspection.effective_digest,
                        component.component_id,
                        component.model_dump_json(),
                    ),
                )
        projection = registry.authority_projection(operation_result=None)
        blocked = next(
            c for c in projection["components"] if c["component_id"] == "mcp:bad"
        )
        assert blocked["support"] == "invalid"
        assert blocked["activation_blockers"] == ["definition_invalid"]


def test_exact_operation_read_and_projection_preserve_guarded_transaction():
    import json

    from tldw_chatbook.Plugins.registry import PluginRegistry

    result = {
        "operation_id": "op",
        "installation_id": "installed",
        "kind": "install",
        "revision_digest": None,
        "result": "committed",
    }
    with PluginRegistry(Path(":memory:")) as registry:
        with pytest.raises(RuntimeError), registry.transaction() as cursor:
            cursor.execute("INSERT INTO installations VALUES ('installed', NULL, 0)")
            cursor.execute(
                "INSERT INTO operations VALUES ('op', 'installed', 'prepared', ?)",
                (json.dumps(result),),
            )
            assert registry.authority_projection(operation_result=result)[
                "installations"
            ]
            assert registry.read_operation("op") == {
                "phase": "prepared",
                "result": result,
            }
            assert registry.read_operation("missing") is None
            raise RuntimeError("rollback owner")
        assert registry.read_operation("op") is None
        assert (
            registry.authority_projection(operation_result=None)["installations"] == []
        )
