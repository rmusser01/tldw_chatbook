"""Semantic catalog and migration behavior with real private SQLite."""

import sqlite3

import pytest

from Tests.Workspaces.project_index_support import (
    make_index_key,
    open_semantic_store,
    page,
)
from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy


def test_fresh_v4_has_exact_semantic_tables_and_read_only_off(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    state = store.load_semantic_state(scope)
    assert store.schema_version == 4
    assert state.semantic_revision == 0
    assert state.enabled is False
    assert state.state == "Off"
    with store.connection() as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM semantic_control").fetchone()[0] == 0
        assert {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE name LIKE 'semantic_%'"
            )
        } >= {"semantic_control", "semantic_generations", "semantic_members"}
    store.close_native()


def test_v3_to_v4_preserves_keyword_policy_frontier_claim_and_rows(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.configure(
        key,
        ProjectIndexPolicy(("src",), (".py",)),
        enabled=True,
        expected_scope=store.load_status(key).scope,
    )
    claim = store.claim_job(
        scope,
        owner_token="native-v1:" + key.binding_namespace + ":test",
        authority_revision=7,
        expected_job_revision=store.load_status(key).job_revision,
    )
    assert store.commit_scan_page(
        scope,
        claim.job_id,
        page(revision=7),
        expected_job_revision=claim.job_revision,
        authority_revision=7,
    )
    before = store.load_status(key)
    with store.transaction() as conn:
        for statement in (
            "DROP TABLE IF EXISTS semantic_control",
            "DROP TABLE IF EXISTS semantic_members",
            "DROP TABLE IF EXISTS semantic_generations",
            "DROP TABLE schema_version",
            "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=3))",
            "INSERT INTO schema_version VALUES(3)",
        ):
            conn.execute(statement)
    store.close_native()
    reopened = open_semantic_store(tmp_path)
    after = reopened.load_status(key)
    assert reopened.schema_version == 4
    assert after.scope == before.scope
    assert (after.indexed, after.manifest_revision, after.job_id) == (
        before.indexed,
        before.manifest_revision,
        before.job_id,
    )
    assert reopened.policy(scope) == ProjectIndexPolicy(("src",), (".py",))
    with reopened.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM directories").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 1
        assert (
            conn.execute("SELECT owner_token FROM active_job").fetchone()[0]
            == claim.owner_token
        )
    reopened.close_native()


def test_v4_ddl_failure_keeps_exact_v3_catalog(tmp_path, monkeypatch):
    store = open_semantic_store(tmp_path)
    path = store.db_path
    with store.transaction() as conn:
        for statement in (
            "DROP TABLE IF EXISTS semantic_control",
            "DROP TABLE IF EXISTS semantic_members",
            "DROP TABLE IF EXISTS semantic_generations",
            "DROP TABLE schema_version",
            "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=3))",
            "INSERT INTO schema_version VALUES(3)",
        ):
            conn.execute(statement)
    store.close_native()
    before = path.read_bytes()
    from tldw_chatbook.DB import Workspace_Project_Index_DB as module

    original = module.WorkspaceProjectIndexDB._migrate

    def fail_after_first(conn, suffix):
        if suffix == "004":
            conn.execute("CREATE TABLE deliberately_partial(x)")
            raise sqlite3.OperationalError("injected DDL failure")
        return original(conn, suffix)

    monkeypatch.setattr(
        module.WorkspaceProjectIndexDB, "_migrate", staticmethod(fail_after_first)
    )
    with pytest.raises(sqlite3.OperationalError, match="injected DDL failure"):
        open_semantic_store(tmp_path)
    assert path.read_bytes() == before
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 3
        assert (
            conn.execute(
                "SELECT name FROM sqlite_schema WHERE name='deliberately_partial'"
            ).fetchone()
            is None
        )


def test_fresh_keyword_status_does_not_create_control_row(tmp_path):
    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    with store.transaction() as conn:
        conn.execute("DELETE FROM control")
    store.close_native()
    before = store.db_path.read_bytes()
    status = store.load_status(key)
    assert status.state == "Off" and status.scope.epoch == 0
    store.close_native()
    assert store.db_path.read_bytes() == before
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM control").fetchone()[0] == 0


def test_v3_cleanup_only_projects_semantic_off_and_clears_without_upgrade(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    store = open_semantic_store(tmp_path)
    key = make_index_key(tmp_path)
    scope = store.load_status(key).scope
    with store.transaction() as conn:
        conn.execute("DROP TABLE semantic_control")
        conn.execute("DROP TABLE semantic_members")
        conn.execute("DROP TABLE semantic_generations")
        conn.execute("DROP TABLE schema_version")
        conn.execute(
            "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=3))"
        )
        conn.execute("INSERT INTO schema_version VALUES(3)")
    path = store.db_path
    store.close_native()
    cleanup = WorkspaceProjectIndexDB(path, cleanup_only=True)
    assert cleanup.schema_version == 3
    assert cleanup.load_semantic_state(scope).state == "Off"
    revoked = cleanup.revoke(key, clear=True, expected_scope=scope)
    assert cleanup.load_status(key).state == "Off"
    assert cleanup.load_status(key).scope == revoked
    with cleanup.connection() as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 3
    cleanup.close_native()


def test_gate_absent_v3_read_refuses_unknown_file_without_recreating_control(tmp_path):
    from tldw_chatbook.DB.Workspace_Project_Index_DB import WorkspaceProjectIndexDB

    store = open_semantic_store(tmp_path)
    with store.transaction() as conn:
        conn.execute("DROP TABLE semantic_members")
        conn.execute("DROP TABLE semantic_generations")
        conn.execute("DROP TABLE semantic_control")
        conn.execute("DROP TABLE schema_version")
        conn.execute(
            "CREATE TABLE schema_version(version INTEGER PRIMARY KEY CHECK(version=3))"
        )
        conn.execute("INSERT INTO schema_version VALUES(3)")
    path = store.db_path
    gate = path.parent.parent / "allocation-v1.lock"
    store.close_native()
    gate.unlink()
    unknown = path.parent / "unexpected.dat"
    unknown.write_bytes(b"no")
    with pytest.raises(ValueError, match="unknown_sidecar"):
        WorkspaceProjectIndexDB(path, cleanup_only=True)
    unknown.unlink()
    legacy = WorkspaceProjectIndexDB(path, cleanup_only=True)
    assert legacy.schema_version == 3
    legacy.close_native()
    assert not gate.exists()
