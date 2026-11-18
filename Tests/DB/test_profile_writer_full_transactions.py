"""Exercise the original SQLite transactions that require process-local evidence."""

import sqlite3
import sys
from contextlib import nullcontext

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows transaction grant")
@pytest.mark.parametrize("borrowed", [False, True], ids=["fresh", "borrowed"])
def test_goal_checkpoint_retires_its_exact_granted_connection(monkeypatch, borrowed):
    from Tests.Agents.test_goal_models import request
    from tldw_chatbook import config
    from tldw_chatbook.Agents.agent_models import RunTerminationReason
    from tldw_chatbook.Agents.goal_models import GoalIterationResult
    from tldw_chatbook.DB import profile_writer, profile_writer_grant
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    profile = config.get_user_data_dir()
    db = AgentRunsDB(profile / "agent_runs.db", reconcile_on_init=False)
    try:
        goal = db.goal_runs.create(request(), launch_id="checkpoint-" + str(borrowed))
        invalid = GoalIterationResult(
            goal.id, 1, "unaccepted", None, None, RunTerminationReason.UNKNOWN_EFFECT
        )
        original_bind = profile_writer_grant._Grant.bind
        original_retire = profile_writer_grant._Grant.retire
        events = []
        statements = []

        def bind(grant, conn):
            original_bind(grant, conn)
            events.append(("bind", conn, grant.prior, grant.synchronous))
            conn.set_trace_callback(statements.append)

        def retire(grant, conn):
            events.append(("retire", conn, conn.in_transaction, conn.execute("PRAGMA synchronous").fetchone()[0]))
            conn.set_trace_callback(None)
            original_retire(grant, conn)

        monkeypatch.setattr(profile_writer_grant._Grant, "bind", bind)
        monkeypatch.setattr(profile_writer_grant._Grant, "retire", retire)
        db.close()
        with db.connection() if borrowed else nullcontext() as prior:
            with pytest.raises(ValueError, match="checkpoint_ownership"):
                db.goal_runs.checkpoint(invalid)
            assert [event[0] for event in events] == ["bind", "retire"]
            _, conn, source, synchronous = events[0]
            assert source is prior
            assert events[1] == ("retire", conn, False, synchronous)
            assert "PRAGMA synchronous=FULL" in statements
            assert "BEGIN IMMEDIATE" in statements
            assert "ROLLBACK" in statements
            if borrowed:
                assert prior is conn and prior.execute("SELECT 1").fetchone()[0] == 1
            else:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    conn.execute("SELECT 1")
        assert db.goal_runs.get(goal.id).checkpoints == ()
    finally:
        db.close()
        profile_writer.drain_profile_writer(profile)


@pytest.mark.parametrize("failed_step", ["commit", "rollback", "restore"])
def test_unknown_native_full_transaction_outcome_is_marked_after_the_effect(failed_step):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    class Receipt:
        ambiguous = False

        def uncertain(self):
            self.ambiguous = True

    class Connection(sqlite3.Connection):
        def commit(self):
            super().commit()
            if failed_step == "commit":
                raise RuntimeError("lost commit")

        def rollback(self):
            super().rollback()
            if failed_step == "rollback":
                raise RuntimeError("lost rollback")

        def execute(self, sql, *args):
            cursor = super().execute(sql, *args)
            if failed_step == "restore" and sql == "PRAGMA synchronous=1":
                raise RuntimeError("lost restore")
            return cursor

    db = AgentRunsDB(":memory:")
    conn = sqlite3.connect(":memory:", factory=Connection)
    try:
        sqlite3.Connection.execute(conn, "PRAGMA synchronous=NORMAL")
        conn.execute("CREATE TABLE effects (id INTEGER)")
        grant = Receipt()
        with pytest.raises(RuntimeError, match="lost " + failed_step):
            with db.automatic_work._full_transaction(conn, grant=grant):
                conn.execute("INSERT INTO effects VALUES (1)")
                if failed_step == "rollback":
                    raise ValueError("original command refused")
        assert grant.ambiguous
        assert not conn.in_transaction
        assert conn.execute("SELECT count(*) FROM effects").fetchone()[0] == (failed_step != "rollback")
    finally:
        conn.close()
        db.close()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows transaction grant")
def test_stock_ledger_admission_rechecks_original_restrictions_and_clock(monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits, AutomaticWorkRefused
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB import automatic_work, profile_writer, profile_writer_grant
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.profile_writer_agent_sources import local_ledger_transaction

    calls = []
    original = profile_writer_grant._local_agent_transaction

    def observed(db, kind):
        calls.append(kind)
        return original(db, kind)

    monkeypatch.setattr(profile_writer_grant, "_local_agent_transaction", observed)
    profile = config.get_user_data_dir()
    db = AgentRunsDB(profile / "agent_runs.db", reconcile_on_init=False)
    ledger = db.automatic_work
    owner = "original-runtime"
    try:
        with pytest.raises(RecoveryRequired, match="profile_writer_grant_source_changed"):
            local_ledger_transaction(ledger, sys._getframe())
        assert ledger.recover(current_owner_id=owner) == 0
        chain = ledger.create_chain("local-chat", root_submission_id="local-submit", limits=AutomaticWorkLimits())
        options = dict(reservation_id="local-attempt", owner_id=owner, kind="generation", amount=1)
        ledger._restrict_chat_start("unconfirmed", owner_id=owner, chain_id=chain)
        with pytest.raises(AutomaticWorkRefused, match="settlement_unconfirmed"):
            ledger.reserve(chain, **options)
        assert ledger.snapshot(chain).reserved["generation"] == 0
        ledger._clear_chat_start_restriction("unconfirmed", owner_id=owner)
        assert ledger.reserve(chain, **options).state == "reserved"
        assert ledger.commit("local-attempt", owner_id=owner)
        with db.connection() as conn:
            assert conn.execute(
                "SELECT clock_owner_id FROM automatic_work_chains WHERE id=?", (chain,)
            ).fetchone()[0] == automatic_work._CLOCK_OWNER_ID
        assert calls == ["ledger.recover", "ledger.reserve", "ledger.reserve", "ledger.commit"]
    finally:
        ledger._clear_chat_start_restriction("unconfirmed", owner_id=owner)
        db.close()
        profile_writer.drain_profile_writer(profile)
