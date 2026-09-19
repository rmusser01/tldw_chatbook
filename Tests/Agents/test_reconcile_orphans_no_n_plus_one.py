"""reconcile_orphaned_runs must batch-hydrate step payloads (TASK-32804.10).

Its terminal-row scan issued one `SELECT payload FROM agent_run_steps WHERE
run_id = ?` per non-running run (295 ms at 250k step rows on first open, inside
BEGIN IMMEDIATE). It now pre-loads every scanned run's steps in one chunked
batched read via the class's existing `_batch_hydrate_steps`.
"""

import json

import pytest

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _seed_terminal_runs(db, n, steps_per_run=3):
    now = "2026-09-19T00:00:00.000Z"
    with db.transaction() as conn:
        for r in range(n):
            run_id = f"run-{r}"
            conn.execute(
                "INSERT INTO agent_runs "
                "(id, conversation_id, agent_kind, status, created_at, updated_at) "
                "VALUES (?, 'conv', 'primary', 'done', ?, ?)",
                (run_id, now, now),
            )
            for s in range(steps_per_run):
                # One step carries the expected terminal kind so the scan sees
                # the run as already captured and writes nothing (steady state).
                payload = {"kind": "agent_run_completed", "owner_seq": 0, "index": s}
                conn.execute(
                    "INSERT INTO agent_run_steps (run_id, seq, payload, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, s, json.dumps(payload), now),
                )


def test_reconcile_batch_hydrates_instead_of_one_query_per_run(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="recon", reconcile_on_init=False)
    try:
        _seed_terminal_runs(db, 5)
        AgentRunsDB._swept_paths.discard(db.db_path_str)  # allow the sweep to run

        calls = {"n": 0, "id_counts": []}
        real = db._batch_hydrate_steps

        def spy(conn, run_ids):
            ids = list(run_ids)
            calls["n"] += 1
            calls["id_counts"].append(len(ids))
            return real(conn, ids)

        db._batch_hydrate_steps = spy  # type: ignore[method-assign]
        try:
            db.reconcile_orphaned_runs()
        finally:
            db._batch_hydrate_steps = real  # type: ignore[method-assign]

        # The terminal-row scan pre-loads all 5 runs in ONE batched call, and
        # run_observations then reads from cache -- so no per-run hydrate.
        assert calls["n"] == 1, calls
        assert calls["id_counts"] == [5], calls
    finally:
        db.close()
