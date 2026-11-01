"""Both the runtime migration and standalone v16->v17 artifact preserve history."""

import sqlite3
from pathlib import Path

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def v16_database(path):
    db = AgentRunsDB(path)
    run_id = db.create_run(conversation_id="legacy", agent_kind="primary")
    db.set_status(run_id, "done", "original result", budget_tokens=23)
    with db.transaction() as conn:
        triggers = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'automatic_%'"
            )
        ]
        for name in triggers:
            # Only schema-owned identifiers read from this newly created DB.
            conn.execute('DROP TRIGGER "' + name.replace('"', '""') + '"')
        conn.execute("DROP TABLE automatic_chat_start_attempts")
        conn.execute("DROP TABLE automatic_wake_claims")
        conn.execute("DROP TABLE automatic_wake_attempts")
        conn.execute("DROP TABLE automatic_work_reservations")
        conn.execute("DROP TABLE IF EXISTS automatic_work_runtime_owner")
        conn.execute("ALTER TABLE agent_runs DROP COLUMN work_chain_id")
        conn.execute("DROP TABLE automatic_work_chains")
        conn.execute("DELETE FROM schema_version WHERE version>=17")
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 16
        )
        assert "work_chain_id" not in {
            row[1] for row in conn.execute("PRAGMA table_info(agent_runs)")
        }
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'automatic_%'"
            ).fetchall()
            == []
        )
    db.close()
    return run_id


@pytest.mark.parametrize("standalone", [False, True])
def test_v16_upgrade_preserves_legacy_rows_without_giving_them_allowance(
    tmp_path, standalone
):
    path = tmp_path / "runs.sqlite"
    run_id = v16_database(path)
    if standalone:
        script = Path(
            "tldw_chatbook/DB/migrations/agent_runs_v16_to_v17_automatic_work.sql"
        ).read_text()
        with sqlite3.connect(path) as conn:
            conn.executescript(script)
            assert (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                == 17
            )
    db = AgentRunsDB(path)
    legacy = db.get_run(run_id)
    assert legacy["result"] == "original result"
    assert legacy["budget_tokens"] == 23
    assert legacy["work_chain_id"] is None
    with db.connection() as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM automatic_work_chains").fetchone()[0]
            == 0
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    chain_id = db.automatic_work.create_chain("new", root_submission_id="new")
    db.close()
    reopened = AgentRunsDB(path)
    assert reopened.automatic_work.snapshot(chain_id).limits.generations == 3
    assert reopened.get_run(run_id)["work_chain_id"] is None
    reopened.close()


def v21_database(path):
    """Remove every v22 artifact and prove the actual migration preconditions."""
    from Tests.DB.test_automatic_work_budget import chain, reserve
    from Tests.DB.test_automatic_wake_attempts import claim, survivor

    db = AgentRunsDB(path)
    root = chain(db, generations=2, budget_tokens=1234)
    run_id = survivor(db, root)
    claim(db, root, [run_id])
    reserve(db, root, "legacy-tokens", kind="tokens", amount=25)
    db.automatic_work.commit("legacy-tokens", owner_id="owner")
    db.automatic_work.settle("legacy-tokens", owner_id="owner", actual_amount=17)
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO agent_definitions(id,name,description,instructions,tool_allowlist,model,enabled,max_wall_seconds,provider,params_json,created_at,updated_at) VALUES ('preserved-preset','Preserved','description','instructions','[]','model-before',1,12.5,'openai','{\"temperature\":0.2}','then','then')"
        )
        conn.execute(
            "UPDATE agent_runs SET resolved_provider='openai',resolved_model='model-before',resolved_base_url='https://example.invalid',resolved_params_json='{\"temperature\":0.2}' WHERE id=?",
            (run_id,),
        )
        conn.execute(
            "INSERT INTO agent_worktrees(run_id,workspace_id,binding_id,locator_fingerprint,repo_root,repo_identity,git_common_dir,git_common_identity,child_path,child_identity,branch,base_sha,execution_id,created_at,updated_at) VALUES (?, 'workspace','binding','locator','repo','repo-identity','common','common-identity','child','child-identity','branch','base','execution','then','then')",
            (run_id,),
        )
        conn.execute("DROP TRIGGER automatic_chat_start_identity_immutable")
        conn.execute("DROP TABLE automatic_chat_start_attempts")
        conn.execute("DROP TRIGGER automatic_chain_root_insert")
        conn.execute("DROP TRIGGER automatic_chain_root_update")
        conn.execute("DROP TRIGGER automatic_chain_identity_immutable")
        conn.execute("DROP INDEX idx_automatic_chains_allowance_root")
        conn.execute(
            "ALTER TABLE automatic_work_chains DROP COLUMN allowance_root_chain_id"
        )
        conn.execute("DELETE FROM schema_version WHERE version=22")
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 21
        )
        assert "allowance_root_chain_id" not in {
            row[1] for row in conn.execute("PRAGMA table_info(automatic_work_chains)")
        }
        assert (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='automatic_chat_start_attempts'"
            ).fetchone()
            is None
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    db.close()
    return root, run_id


@pytest.mark.parametrize("standalone", [False, True])
def test_v21_upgrade_preserves_roots_reservations_and_wake_scope(tmp_path, standalone):
    from Tests.DB.test_automatic_chat_starts import prepare

    path = tmp_path / "runs.sqlite"
    root, run_id = v21_database(path)
    tables = (
        "agent_definitions",
        "agent_runs",
        "agent_worktrees",
        "automatic_work_chains",
        "automatic_work_reservations",
        "automatic_wake_attempts",
        "automatic_wake_claims",
    )

    def rows():
        with sqlite3.connect(path) as connection:
            connection.row_factory = sqlite3.Row
            result = {}
            for table in tables:
                result[table] = [
                    dict(row)
                    for row in connection.execute(
                        f"SELECT * FROM {table} ORDER BY rowid"
                    )
                ]
                for row in result[table]:
                    row.pop("allowance_root_chain_id", None)
            return result

    predecessor = rows()
    if standalone:
        script = Path(
            "tldw_chatbook/DB/migrations/agent_runs_v21_to_v22_chat_starts.sql"
        ).read_text()
        with sqlite3.connect(path) as conn:
            conn.executescript(script)
            assert (
                conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
                == 22
            )
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
            # Exercise the standalone guards before runtime opening can repair
            # a missing schema artifact and conceal a broken SQL migration.
            conn.execute(
                "INSERT INTO automatic_work_chains (id, conversation_id, root_submission_id, limits_json, created_at, last_observed_at, allowance_root_chain_id) "
                "SELECT 'standalone-child','target','standalone-child',limits_json,created_at,last_observed_at,id FROM automatic_work_chains WHERE id=?",
                (root,),
            )
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute(
                    "UPDATE automatic_work_chains SET allowance_root_chain_id=NULL WHERE id='standalone-child'"
                )
            with pytest.raises(sqlite3.IntegrityError, match="direct root"):
                conn.execute(
                    "INSERT INTO automatic_work_chains (id, conversation_id, root_submission_id, limits_json, created_at, last_observed_at, allowance_root_chain_id) "
                    "SELECT 'invalid','target','invalid',limits_json,created_at,last_observed_at,'standalone-child' FROM automatic_work_chains WHERE id=?",
                    (root,),
                )
            conn.rollback()
    for _ in range(2):
        reopened = AgentRunsDB(path)
        assert rows() == predecessor
        reopened.close()
    db = AgentRunsDB(path)
    try:
        snapshot = db.automatic_work.snapshot(root)
        assert snapshot.limits.generations == 2
        assert snapshot.limits.budget_tokens == 1234
        assert snapshot.used["tokens"] == 17
        assert snapshot.reserved["generation"] == 1
        assert db.automatic_work.allowance_root(root) == root
        assert db.get_run(run_id)["work_chain_id"] == root
        assert (
            db.automatic_work.read_attempt("attempt", owner_id="owner").state
            == "prepared"
        )
        source = db.create_run(
            conversation_id="conversation", agent_kind="primary", work_chain_id=root
        )
        first = prepare(db, source, "target", "first")
        assert db.automatic_work.allowance_root(first.chain_id) == root
        with db.transaction() as conn:
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                conn.execute(
                    "UPDATE automatic_work_chains SET allowance_root_chain_id=NULL WHERE id=?",
                    (first.chain_id,),
                )
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        db.close()
