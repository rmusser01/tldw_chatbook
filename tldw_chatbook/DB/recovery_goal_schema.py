"""Exact goal catalog delta and installed staging authority (ADR-227)."""

GOAL_OBJECTS = (
    "CREATE TABLE goal_checkpoints (\n    attempt_id TEXT PRIMARY KEY REFERENCES automatic_wake_attempts(id),\n    goal_id TEXT NOT NULL REFERENCES goal_runs(id),\n    payload_hash TEXT NOT NULL,\n    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072)\n)",
    "CREATE TABLE goal_evidence (\n    id TEXT PRIMARY KEY,\n    goal_id TEXT NOT NULL REFERENCES goal_runs(id),\n    attempt_id TEXT NOT NULL REFERENCES automatic_wake_attempts(id),\n    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072)\n)",
    "CREATE TABLE goal_iterations (\n    id TEXT PRIMARY KEY,\n    goal_id TEXT NOT NULL REFERENCES goal_runs(id),\n    ordinal INTEGER NOT NULL CHECK (ordinal > 0),\n    launch_id TEXT NOT NULL UNIQUE,\n    attempt_id TEXT NOT NULL UNIQUE REFERENCES automatic_wake_attempts(id),\n    run_id TEXT REFERENCES agent_runs(id),\n    status TEXT NOT NULL,\n    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),\n    check_results_json TEXT NOT NULL DEFAULT '[]' CHECK (length(CAST(check_results_json AS BLOB)) <= 131072),\n    evidence_refs_json TEXT NOT NULL DEFAULT '[]' CHECK (length(CAST(evidence_refs_json AS BLOB)) <= 8192),\n    UNIQUE(goal_id, ordinal)\n)",
    "CREATE TABLE goal_payload_reservations (\n    attempt_id TEXT PRIMARY KEY REFERENCES automatic_wake_attempts(id),\n    goal_id TEXT NOT NULL REFERENCES goal_runs(id),\n    bytes INTEGER NOT NULL CHECK(bytes >= 0)\n)",
    "CREATE TABLE goal_reports (\n    id TEXT PRIMARY KEY,\n    goal_id TEXT NOT NULL REFERENCES goal_runs(id),\n    iteration_id TEXT UNIQUE REFERENCES goal_iterations(id),\n    payload_json TEXT NOT NULL CHECK (length(CAST(payload_json AS BLOB)) <= 65536)\n)",
    "CREATE TABLE goal_runs (\n    id TEXT PRIMARY KEY,\n    launch_id TEXT NOT NULL UNIQUE,\n    payload_hash TEXT NOT NULL,\n    request_json TEXT NOT NULL CHECK (length(CAST(request_json AS BLOB)) <= 131072),\n    conversation_id TEXT NOT NULL UNIQUE,\n    chain_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_chains(id),\n    revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),\n    status TEXT NOT NULL DEFAULT 'starting',\n    pause_reason TEXT,\n    created_at REAL NOT NULL,\n    updated_at REAL NOT NULL\n)",
    "CREATE TABLE goal_waits (\n    goal_id TEXT PRIMARY KEY REFERENCES goal_runs(id),\n    retry_at REAL NOT NULL,\n    reason TEXT NOT NULL\n)",
    "CREATE TRIGGER goal_launch_immutable\nBEFORE UPDATE OF launch_id, payload_hash, request_json, conversation_id, chain_id ON goal_runs\nWHEN OLD.launch_id IS NOT NEW.launch_id OR OLD.payload_hash IS NOT NEW.payload_hash\n OR (OLD.request_json IS NOT NEW.request_json AND NOT (NEW.request_json='' AND NEW.status='removed'\n     AND OLD.status IN ('completed','paused','awaiting_result_review','stopped')\n     AND NOT EXISTS (SELECT 1 FROM goal_iterations i JOIN automatic_wake_attempts a ON a.id=i.attempt_id\n         WHERE i.goal_id=OLD.id AND a.state IN ('prepared','accepted','review_required'))))\n OR OLD.conversation_id IS NOT NEW.conversation_id OR OLD.chain_id IS NOT NEW.chain_id\nBEGIN SELECT RAISE(ABORT, 'goal launch is immutable'); END",
)

GOAL_AUTO_INDEXES = frozenset(
    {
        ("sqlite_autoindex_goal_iterations_4", "goal_iterations"),
        ("sqlite_autoindex_goal_reports_2", "goal_reports"),
        ("sqlite_autoindex_goal_runs_3", "goal_runs"),
        ("sqlite_autoindex_goal_reports_1", "goal_reports"),
        ("sqlite_autoindex_goal_iterations_2", "goal_iterations"),
        ("sqlite_autoindex_goal_iterations_1", "goal_iterations"),
        ("sqlite_autoindex_goal_runs_4", "goal_runs"),
        ("sqlite_autoindex_goal_evidence_1", "goal_evidence"),
        ("sqlite_autoindex_goal_runs_1", "goal_runs"),
        ("sqlite_autoindex_goal_runs_2", "goal_runs"),
        ("sqlite_autoindex_goal_checkpoints_1", "goal_checkpoints"),
        ("sqlite_autoindex_goal_waits_1", "goal_waits"),
        ("sqlite_autoindex_goal_iterations_3", "goal_iterations"),
        ("sqlite_autoindex_goal_payload_reservations_1", "goal_payload_reservations"),
    }
)

GOAL_TABLES = frozenset(
    {
        "goal_runs",
        "goal_reports",
        "goal_checkpoints",
        "goal_iterations",
        "goal_evidence",
        "goal_waits",
        "goal_payload_reservations",
    }
)

GOAL_ATTEMPT_COLUMN = "attempt_kind TEXT NOT NULL DEFAULT 'fleet_wake' CHECK (attempt_kind IN ('fleet_wake', 'goal_iteration'))"


def goal_catalog(catalog):
    """Apply only the qualified v23-to-v24 constructor delta."""
    import re

    rows = (
        tuple(
            sql[:-1] + ", " + GOAL_ATTEMPT_COLUMN + ")"
            if sql.startswith("CREATE TABLE automatic_wake_attempts ")
            else sql
            for sql in catalog
        )
        + GOAL_OBJECTS
    )

    def key(sql):
        match = re.match(
            r'CREATE (?:UNIQUE )?(TABLE|INDEX|TRIGGER|VIEW) ["`]?([^\s"`(]+)', sql
        )
        return match[1].lower(), match[2]

    return tuple(sorted(rows, key=key))


GOAL_MIGRATION = (
    "ALTER TABLE automatic_wake_attempts ADD COLUMN " + GOAL_ATTEMPT_COLUMN,
    *GOAL_OBJECTS,
    "INSERT INTO schema_version(version) VALUES (24)",
)


def validate_goal_rows(connection):
    """Validate exact goal payloads and cross-table ownership without opening sources."""
    import hashlib
    from typing import get_args

    from tldw_chatbook.Agents.goal_models import (
        GoalCheckpoint,
        GoalEvidence,
        GoalReport,
        GoalRequest,
        GoalSnapshot,
    )

    # Goal rows exist from v24 on (v25 adds review-boundary claims only).
    if connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] < 24:
        return ()
    invalid = connection.execute(
        "SELECT 1 FROM goal_runs g JOIN automatic_work_chains c ON c.id=g.chain_id "
        "WHERE c.conversation_id!=g.conversation_id OR c.allowance_root_chain_id IS NOT NULL "
        "UNION ALL SELECT 1 FROM goal_iterations i JOIN goal_runs g ON g.id=i.goal_id "
        "JOIN automatic_wake_attempts a ON a.id=i.attempt_id "
        "WHERE a.attempt_kind!='goal_iteration' OR a.chain_id!=g.chain_id OR a.conversation_id!=g.conversation_id "
        "OR (i.run_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM agent_runs r WHERE r.id=i.run_id AND r.work_chain_id=g.chain_id AND r.conversation_id=g.conversation_id)) "
        "UNION ALL SELECT 1 FROM automatic_wake_attempts a WHERE a.attempt_kind='goal_iteration' "
        "AND NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=a.id) "
        "UNION ALL SELECT 1 FROM goal_reports r JOIN goal_iterations i ON i.id=r.iteration_id WHERE r.goal_id!=i.goal_id "
        "UNION ALL SELECT 1 FROM goal_checkpoints c WHERE NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=c.attempt_id AND i.goal_id=c.goal_id) "
        "UNION ALL SELECT 1 FROM goal_evidence e WHERE NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=e.attempt_id AND i.goal_id=e.goal_id) "
        "UNION ALL SELECT 1 FROM goal_payload_reservations p WHERE NOT EXISTS (SELECT 1 FROM goal_iterations i WHERE i.attempt_id=p.attempt_id AND i.goal_id=p.goal_id) LIMIT 1"
    ).fetchone()
    if invalid is not None:
        return ("invalid_goal_ownership",)
    try:
        for status, payload, digest in connection.execute(
            "SELECT status, request_json, payload_hash FROM goal_runs"
        ):
            if status not in get_args(GoalSnapshot.model_fields["status"].annotation):
                return ("invalid_goal_payload",)
            if status == "removed":
                if payload:
                    return ("invalid_goal_payload",)
                continue
            GoalRequest.model_validate_json(payload)
            if hashlib.sha256(payload.encode()).hexdigest() != digest:
                return ("invalid_goal_payload",)
        for (payload,) in connection.execute("SELECT payload_json FROM goal_reports"):
            GoalReport.model_validate_json(payload)
        for attempt_id, ordinal, payload in connection.execute(
            "SELECT c.attempt_id,i.ordinal,c.payload_json FROM goal_checkpoints c "
            "JOIN goal_iterations i ON i.attempt_id=c.attempt_id"
        ):
            checkpoint = GoalCheckpoint.model_validate_json(payload)
            if checkpoint.id != attempt_id or checkpoint.ordinal != ordinal:
                return ("invalid_goal_ownership",)
        for evidence_id, goal_id, attempt_id, run_id, payload in connection.execute(
            "SELECT e.id,e.goal_id,e.attempt_id,i.run_id,e.payload_json FROM goal_evidence e "
            "JOIN goal_iterations i ON i.attempt_id=e.attempt_id"
        ):
            evidence = GoalEvidence.model_validate_json(payload)
            if (
                evidence.id,
                evidence.goal_id,
                evidence.attempt_id,
                evidence.run_id,
            ) != (evidence_id, goal_id, attempt_id, run_id):
                return ("invalid_goal_ownership",)
    except (ValueError, TypeError):
        return ("invalid_goal_payload",)
    return ()
