"""task-18601 part A: agent_runs.steps moved out of a single JSON blob
column into a child table (``agent_run_steps``), with a compatibility
read path for runs written before this change.

The defect this guards against: ``AgentRunsDB.append_steps`` used to
read the ENTIRE step-log blob, ``json.loads`` it, extend the list,
``json.dumps`` it, and rewrite the whole column -- O(n) work per call,
O(n^2) over a run's lifetime. Measured on a real DB (~200-byte steps):
append #1 cost 0.05ms, append #500 cost 0.49ms, append #2000 cost
2.18ms (44x the first) -- ~5.4 minutes of write churn extrapolated to
the 25,000-step budget AC#1 names for a single run.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from tldw_chatbook.DB import AgentRuns_DB as agent_runs_db
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "agent_runs.db", client_id="test")


# --- Scaling: the defect itself ---------------------------------------


def test_append_steps_cost_does_not_scale_with_log_size(db):
    """Per-append cost must not grow with the run's existing step count.

    Batched (50 calls per measurement) to smooth per-call timer noise --
    a single ~sub-millisecond call is too close to timer resolution to
    compare reliably in isolation. Compares a batch of appends taken
    EARLY in the run's life against a batch taken LATE (after ~2000
    prior steps), with a generous margin so this doesn't flake on a
    loaded machine: the regression under test showed a 44x per-call
    slowdown by append #2000, so even heavy scheduling/GC/disk noise
    should leave a wide gap under a 5x threshold for a correctly O(1)
    (amortized) implementation.
    """
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    step = {"index": 0, "kind": "model", "summary": "x" * 150}

    def _time_batch(n: int) -> float:
        start = time.perf_counter()
        for _ in range(n):
            db.append_steps(run_id, [step])
        return time.perf_counter() - start

    # Warm up (first-open costs, page cache, WAL growth) before timing.
    _time_batch(50)
    early = _time_batch(50)
    # Grow the log substantially before the second measurement -- this
    # is exactly the range (~2000 prior steps) the measured regression
    # showed 44x cost growth over.
    _time_batch(1900)
    late = _time_batch(50)

    assert late < early * 5 + 0.05, (
        "append_steps cost grew with the run's existing step count: "
        f"an early batch (~50-100 prior steps) took {early:.4f}s, a "
        f"late batch (~2000-2050 prior steps) took {late:.4f}s -- this "
        "looks like a regression back to a read-modify-write of the "
        "whole step log"
    )


# --- Dual read: legacy blob + new child-table rows ----------------------


def test_legacy_blob_steps_still_readable(db):
    """A run written in the old (pre-child-table) format -- steps living
    only in the ``agent_runs.steps`` JSON blob, no ``agent_run_steps``
    rows -- must still read back correctly. Inserts the blob directly
    (bypassing ``append_steps``) to simulate a run persisted before this
    change landed."""
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    legacy_steps = [
        {"index": 0, "kind": "model", "summary": "legacy hello"},
        {"index": 1, "kind": "tool_call", "tool_name": "calculator"},
    ]
    with db.transaction() as conn:
        conn.execute(
            "UPDATE agent_runs SET steps = ? WHERE id = ?",
            (json.dumps(legacy_steps), run_id),
        )

    run = db.get_run(run_id)
    assert run["steps"] == legacy_steps


def test_mixed_legacy_blob_and_new_appends_return_in_order(db):
    """A run that has BOTH legacy blob steps AND new child-table rows
    (a legacy run appended to again after this change) must return
    every step, blob steps first, then appended rows in append order."""
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    legacy_steps = [{"index": 0, "kind": "model", "summary": "legacy"}]
    with db.transaction() as conn:
        conn.execute(
            "UPDATE agent_runs SET steps = ? WHERE id = ?",
            (json.dumps(legacy_steps), run_id),
        )

    db.append_steps(
        run_id, [{"index": 1, "kind": "tool_call", "tool_name": "calc"}]
    )
    db.append_steps(run_id, [{"index": 2, "kind": "model", "summary": "done"}])

    steps = db.get_run(run_id)["steps"]
    assert [s["index"] for s in steps] == [0, 1, 2]
    assert steps[0]["summary"] == "legacy"
    assert steps[1]["tool_name"] == "calc"
    assert steps[2]["summary"] == "done"


def test_fresh_run_has_no_blob_steps_to_parse(db):
    """A run created after this change never has legacy blob steps --
    its whole history lives in ``agent_run_steps``, and hydration is a
    single indexed child-table SELECT."""
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.append_steps(run_id, [{"index": 0, "kind": "model", "summary": "hi"}])
    db.append_steps(
        run_id, [{"index": 1, "kind": "tool_call", "tool_name": "calculator"}]
    )

    with db.connection() as conn:
        blob = conn.execute(
            "SELECT steps FROM agent_runs WHERE id = ?", (run_id,)
        ).fetchone()["steps"]
    assert blob == "[]"

    steps = db.get_run(run_id)["steps"]
    assert [s["index"] for s in steps] == [0, 1]
    assert steps[1]["tool_name"] == "calculator"


# --- Cascade -------------------------------------------------------------


def test_deleting_run_cascades_to_step_rows(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.append_steps(run_id, [{"index": 0, "kind": "model"}])
    db.append_steps(run_id, [{"index": 1, "kind": "model"}])

    with db.connection() as conn:
        before = conn.execute(
            "SELECT COUNT(*) AS n FROM agent_run_steps WHERE run_id = ?",
            (run_id,),
        ).fetchone()["n"]
    assert before == 2

    with db.transaction() as conn:
        conn.execute("DELETE FROM agent_runs WHERE id = ?", (run_id,))

    with db.connection() as conn:
        after = conn.execute(
            "SELECT COUNT(*) AS n FROM agent_run_steps WHERE run_id = ?",
            (run_id,),
        ).fetchone()["n"]
    assert after == 0


# --- Metadata-only read (AC#2) -------------------------------------------


def test_get_run_metadata_omits_steps_and_returns_other_fields(db):
    run_id = db.create_run(
        conversation_id="c", agent_kind="primary", budget={"max_steps": 5}
    )
    db.append_steps(run_id, [{"index": 0, "kind": "model", "summary": "hi"}])
    db.set_status(run_id, "done", result="the answer")

    meta = db.get_run_metadata(run_id)
    assert meta is not None
    assert "steps" not in meta
    assert meta["status"] == "done"
    assert meta["result"] == "the answer"
    assert meta["budget"] == {"max_steps": 5}
    assert meta["conversation_id"] == "c"
    assert meta["agent_kind"] == "primary"


def test_get_run_metadata_missing_run_returns_none(db):
    assert db.get_run_metadata("nope") is None


def test_latest_primary_run_metadata_omits_steps(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    db.append_steps(run_id, [{"index": 0, "kind": "model"}])

    meta = db.latest_primary_run_metadata("c")
    assert meta is not None
    assert meta["id"] == run_id
    assert "steps" not in meta


def test_append_steps_unknown_run_id_raises(db):
    with pytest.raises(KeyError):
        db.append_steps("nope", [{"index": 0, "kind": "model"}])


# --- Chunking: SQLite's host-parameter ceiling ------------------------


def test_batch_hydrate_survives_more_run_ids_than_the_old_param_ceiling(db):
    """Hydrating many runs at once must not hit SQLite's variable limit.

    ``_batch_hydrate_steps`` binds one parameter per run id, and no
    caller bounds the list -- ``ConsoleAgentController.subagent_runs``
    asks for every run in a conversation. The ceiling is BUILD-dependent:
    32766 on SQLite >= 3.32 but 999 on older builds, and this project's
    floor is Python 3.11, which can ship either. So this test lowers the
    connection's own limit to the old value rather than trusting the
    local build -- otherwise it would pass unchunked on a modern SQLite
    and prove nothing about the machines that actually break.
    """
    run_ids = [
        db.create_run(conversation_id="c", agent_kind="primary")
        for _ in range(5)
    ]
    for i, rid in enumerate(run_ids):
        db.append_steps(rid, [{"index": 0, "kind": "model", "summary": str(i)}])

    # Pad to well past the old ceiling with ids that have no rows: the
    # bound-parameter count is what overflows, not the row count.
    padded = [f"absent-{n}" for n in range(1200)] + run_ids

    with db.transaction() as conn:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        grouped = db._batch_hydrate_steps(conn, padded)

    assert set(grouped) == set(run_ids), "every run with rows must hydrate"
    for i, rid in enumerate(run_ids):
        assert [s["summary"] for s in grouped[rid]] == [str(i)]


def test_batch_hydrate_groups_correctly_across_chunk_boundaries(monkeypatch):
    """Splitting the read into chunks must not drop or mis-group rows.

    Run ids are grouped per chunk and merged; a boundary bug (last chunk
    overwriting instead of extending, or a run's rows split across two
    chunks) is invisible at the real chunk size, so shrink it to 2.
    """
    monkeypatch.setattr(agent_runs_db, "_IN_CLAUSE_CHUNK", 2)
    with tempfile.TemporaryDirectory() as tmp:
        db = AgentRunsDB(Path(tmp) / "chunks.db", client_id="test")
        run_ids = [
            db.create_run(conversation_id="c", agent_kind="primary")
            for _ in range(7)
        ]
        for i, rid in enumerate(run_ids):
            db.append_steps(
                rid,
                [
                    {"index": 0, "kind": "model", "summary": f"{i}-a"},
                    {"index": 1, "kind": "model", "summary": f"{i}-b"},
                ],
            )

        with db.transaction() as conn:
            grouped = db._batch_hydrate_steps(conn, run_ids)

        assert set(grouped) == set(run_ids)
        for i, rid in enumerate(run_ids):
            assert [s["summary"] for s in grouped[rid]] == [f"{i}-a", f"{i}-b"]
