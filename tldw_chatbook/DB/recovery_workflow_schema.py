"""Frozen v25 workflow catalog and private staging validation (ADR-285)."""

import sqlite3

WORKFLOW_OBJECTS = (
    "CREATE INDEX idx_workflow_conversation ON workflow_runs(conversation_id, created_at DESC, id)",
    "CREATE TABLE workflow_active_intervals (\n    id TEXT PRIMARY KEY,\n    workflow_id TEXT NOT NULL REFERENCES workflow_runs(id),\n    owner_id TEXT NOT NULL,\n    started_at REAL NOT NULL CHECK(started_at>=0),\n    ended_at REAL CHECK(ended_at>=started_at),\n    charged_seconds REAL NOT NULL DEFAULT 0 CHECK(charged_seconds>=0)\n)",
    "CREATE TABLE workflow_decisions (\n    id TEXT PRIMARY KEY,\n    workflow_id TEXT NOT NULL,\n    plan_revision INTEGER NOT NULL,\n    phase_id TEXT NOT NULL,\n    control_revision INTEGER NOT NULL CHECK(control_revision>0),\n    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072),\n    UNIQUE(workflow_id, control_revision),\n    FOREIGN KEY(workflow_id, plan_revision, phase_id) REFERENCES workflow_phases(workflow_id, plan_revision, phase_id)\n)",
    "CREATE TABLE workflow_payload_reservations (\n    attempt_id TEXT PRIMARY KEY,\n    workflow_id TEXT NOT NULL,\n    bytes INTEGER NOT NULL CHECK(typeof(bytes)='integer' AND bytes BETWEEN 1 AND 131072),\n    state TEXT NOT NULL CHECK(state IN ('reserved','published','removed')),\n    FOREIGN KEY(attempt_id, workflow_id) REFERENCES workflow_task_attempts(id, workflow_id)\n)",
    "CREATE TABLE workflow_phase_attempts (\n    id TEXT PRIMARY KEY,\n    workflow_id TEXT NOT NULL,\n    plan_revision INTEGER NOT NULL,\n    phase_id TEXT NOT NULL,\n    ordinal INTEGER NOT NULL CHECK(typeof(ordinal)='integer' AND ordinal>0),\n    native_attempt_id TEXT UNIQUE REFERENCES automatic_wake_attempts(id),\n    status TEXT NOT NULL CHECK(status IN ('prepared','accepted','completed','failed','uncertain','closed')),\n    retirement TEXT NOT NULL DEFAULT 'pending' CHECK(retirement IN ('pending','drained','unknown')),\n    UNIQUE(workflow_id, plan_revision, phase_id, ordinal),\n    FOREIGN KEY(workflow_id, plan_revision, phase_id) REFERENCES workflow_phases(workflow_id, plan_revision, phase_id)\n)",
    "CREATE TABLE workflow_phases (\n    workflow_id TEXT NOT NULL,\n    plan_revision INTEGER NOT NULL,\n    phase_id TEXT NOT NULL,\n    ordinal INTEGER NOT NULL CHECK(ordinal BETWEEN 1 AND 6),\n    PRIMARY KEY(workflow_id, plan_revision, phase_id),\n    UNIQUE(workflow_id, plan_revision, ordinal),\n    FOREIGN KEY(workflow_id, plan_revision) REFERENCES workflow_plan_revisions(workflow_id, revision)\n)",
    "CREATE TABLE workflow_plan_revisions (\n    workflow_id TEXT NOT NULL REFERENCES workflow_runs(id),\n    revision INTEGER NOT NULL CHECK(typeof(revision)='integer' AND revision>0),\n    payload_hash TEXT NOT NULL,\n    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072),\n    PRIMARY KEY(workflow_id, revision)\n)",
    "CREATE TABLE workflow_results (\n    attempt_id TEXT PRIMARY KEY REFERENCES workflow_payload_reservations(attempt_id),\n    workflow_id TEXT NOT NULL,\n    payload_hash TEXT NOT NULL,\n    payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072),\n    FOREIGN KEY(attempt_id, workflow_id) REFERENCES workflow_task_attempts(id, workflow_id)\n)",
    "CREATE TABLE workflow_runs (\n    id TEXT PRIMARY KEY,\n    launch_id TEXT NOT NULL UNIQUE,\n    payload_hash TEXT NOT NULL,\n    request_json TEXT NOT NULL CHECK(length(CAST(request_json AS BLOB)) <= 131072),\n    conversation_id TEXT NOT NULL,\n    chain_id TEXT NOT NULL UNIQUE REFERENCES automatic_work_chains(id),\n    control_revision INTEGER NOT NULL DEFAULT 1 CHECK(typeof(control_revision)='integer' AND control_revision>0),\n    progress_cursor INTEGER NOT NULL DEFAULT 0 CHECK(typeof(progress_cursor)='integer' AND progress_cursor>=0),\n    status TEXT NOT NULL DEFAULT 'awaiting_plan' CHECK(status IN ('awaiting_plan','awaiting_approval','running','paused','stopped','completed','closed','recovery_required','removed','awaiting_resume')),\n    created_at REAL NOT NULL,\n    updated_at REAL NOT NULL\n)",
    "CREATE TABLE workflow_task_attempts (\n    id TEXT PRIMARY KEY,\n    workflow_id TEXT NOT NULL,\n    plan_revision INTEGER NOT NULL,\n    phase_id TEXT NOT NULL,\n    task_id TEXT NOT NULL,\n    ordinal INTEGER NOT NULL CHECK(typeof(ordinal)='integer' AND ordinal>0),\n    run_id TEXT UNIQUE REFERENCES agent_runs(id),\n    native_attempt_id TEXT UNIQUE REFERENCES automatic_wake_attempts(id),\n    status TEXT NOT NULL CHECK(status IN ('prepared','accepted','completed','failed','uncertain','closed')),\n    retirement TEXT NOT NULL DEFAULT 'pending' CHECK(retirement IN ('pending','drained','unknown')),\n    UNIQUE(id, workflow_id),\n    UNIQUE(workflow_id, plan_revision, task_id, ordinal),\n    FOREIGN KEY(workflow_id, plan_revision, phase_id, task_id) REFERENCES workflow_tasks(workflow_id, plan_revision, phase_id, task_id)\n)",
    "CREATE TABLE workflow_tasks (\n    workflow_id TEXT NOT NULL,\n    plan_revision INTEGER NOT NULL,\n    phase_id TEXT NOT NULL,\n    task_id TEXT NOT NULL,\n    ordinal INTEGER NOT NULL CHECK(ordinal BETWEEN 1 AND 12),\n    PRIMARY KEY(workflow_id, plan_revision, phase_id, task_id),\n    UNIQUE(workflow_id, plan_revision, task_id),\n    FOREIGN KEY(workflow_id, plan_revision, phase_id) REFERENCES workflow_phases(workflow_id, plan_revision, phase_id)\n)",
    "CREATE TRIGGER workflow_decision_immutable\nBEFORE UPDATE ON workflow_decisions\nBEGIN SELECT RAISE(ABORT, 'workflow decision is immutable'); END",
    "CREATE TRIGGER workflow_launch_immutable\nBEFORE UPDATE OF id, launch_id, payload_hash, request_json, conversation_id, chain_id ON workflow_runs\nBEGIN SELECT RAISE(ABORT, 'workflow launch is immutable'); END",
    "CREATE TRIGGER workflow_plan_immutable\nBEFORE UPDATE ON workflow_plan_revisions\nBEGIN SELECT RAISE(ABORT, 'workflow plan is immutable'); END",
    "CREATE TRIGGER workflow_result_immutable\nBEFORE UPDATE ON workflow_results\nBEGIN SELECT RAISE(ABORT, 'workflow result is immutable'); END",
)

WORKFLOW_TABLES = frozenset(
    [
        "workflow_active_intervals",
        "workflow_decisions",
        "workflow_payload_reservations",
        "workflow_phase_attempts",
        "workflow_phases",
        "workflow_plan_revisions",
        "workflow_results",
        "workflow_runs",
        "workflow_task_attempts",
        "workflow_tasks",
    ]
)
WORKFLOW_INDEXES = frozenset(
    [
        ("idx_workflow_conversation", "workflow_runs"),
        ("sqlite_autoindex_workflow_active_intervals_1", "workflow_active_intervals"),
        ("sqlite_autoindex_workflow_decisions_1", "workflow_decisions"),
        ("sqlite_autoindex_workflow_decisions_2", "workflow_decisions"),
        (
            "sqlite_autoindex_workflow_payload_reservations_1",
            "workflow_payload_reservations",
        ),
        ("sqlite_autoindex_workflow_phase_attempts_1", "workflow_phase_attempts"),
        ("sqlite_autoindex_workflow_phase_attempts_2", "workflow_phase_attempts"),
        ("sqlite_autoindex_workflow_phase_attempts_3", "workflow_phase_attempts"),
        ("sqlite_autoindex_workflow_phases_1", "workflow_phases"),
        ("sqlite_autoindex_workflow_phases_2", "workflow_phases"),
        ("sqlite_autoindex_workflow_plan_revisions_1", "workflow_plan_revisions"),
        ("sqlite_autoindex_workflow_results_1", "workflow_results"),
        ("sqlite_autoindex_workflow_runs_1", "workflow_runs"),
        ("sqlite_autoindex_workflow_runs_2", "workflow_runs"),
        ("sqlite_autoindex_workflow_runs_3", "workflow_runs"),
        ("sqlite_autoindex_workflow_task_attempts_1", "workflow_task_attempts"),
        ("sqlite_autoindex_workflow_task_attempts_2", "workflow_task_attempts"),
        ("sqlite_autoindex_workflow_task_attempts_3", "workflow_task_attempts"),
        ("sqlite_autoindex_workflow_task_attempts_4", "workflow_task_attempts"),
        ("sqlite_autoindex_workflow_task_attempts_5", "workflow_task_attempts"),
        ("sqlite_autoindex_workflow_tasks_1", "workflow_tasks"),
        ("sqlite_autoindex_workflow_tasks_2", "workflow_tasks"),
    ]
)
WORKFLOW_TRIGGERS = frozenset(
    [
        ("workflow_decision_immutable", "workflow_decisions"),
        ("workflow_launch_immutable", "workflow_runs"),
        ("workflow_plan_immutable", "workflow_plan_revisions"),
        ("workflow_result_immutable", "workflow_results"),
    ]
)

WORKFLOW_MIGRATION = (
    *sorted(WORKFLOW_OBJECTS, key=lambda sql: not sql.startswith("CREATE TABLE")),
    "INSERT INTO schema_version(version) VALUES (25)",
)


def workflow_catalog(catalog: tuple[str, ...]) -> tuple[str, ...]:
    """Append only the captured, constructor-qualified v24-to-v25 delta."""
    import re

    def key(sql):
        match = re.match(
            r'CREATE (?:UNIQUE )?(TABLE|INDEX|TRIGGER|VIEW) ["`]?([^\s"`(]+)', sql
        )
        return match[1].lower(), match[2]

    return tuple(sorted((*catalog, *WORKFLOW_OBJECTS), key=key))


def validate_workflow_rows(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Validate passive rows and exact ownership without probing any captured path."""
    import hashlib
    import json
    import math
    from dataclasses import asdict

    from tldw_chatbook.Agents.workflow_models import (
        WorkflowDecision,
        WorkflowPlan,
        WorkflowRequest,
        WorkflowTaskAttempt,
        WorkflowTaskResult,
    )

    from .workflow_runs import WorkflowRunsStore

    if (
        connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        >= 27
    ):
        from tldw_chatbook.Agents.workflow_models import WorkflowCheckpoint

        for attempt_id, workflow_id, payload_hash, payload in connection.execute(
            "SELECT * FROM workflow_checkpoints"
        ):
            try:
                checkpoint = WorkflowCheckpoint.model_validate_json(payload)
                row = connection.execute(
                    "SELECT r.payload_hash,r.payload_json,p.bytes FROM workflow_results r JOIN workflow_payload_reservations p ON p.attempt_id=r.attempt_id WHERE r.attempt_id=? AND r.workflow_id=?",
                    (attempt_id, workflow_id),
                ).fetchone()
                if (
                    row is None
                    or checkpoint.attempt_id != attempt_id
                    or checkpoint.workflow_id != workflow_id
                    or hashlib.sha256(payload.encode()).hexdigest() != payload_hash
                    or checkpoint.result_digest != row[0]
                    or checkpoint.run_id
                    != WorkflowTaskResult.model_validate_json(row[1]).run_id
                    or any(
                        connection.execute(
                            "SELECT 1 FROM change_snapshots WHERE id=? AND run_id=?",
                            (ref, checkpoint.run_id),
                        ).fetchone()
                        is None
                        for ref in checkpoint.change_window_refs
                    )
                    or len(payload.encode()) + len(row[1].encode()) > row[2]
                ):
                    return ("invalid_workflow_checkpoint",)
            except (ValueError, TypeError):
                return ("invalid_workflow_checkpoint",)

    if connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] < 25:
        return ()
    previous = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        if (
            connection.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
            >= 26
        ):
            invalid = connection.execute(
                "SELECT 1 FROM automatic_wake_attempts a LEFT JOIN workflow_phase_attempts p ON p.native_attempt_id=a.id "
                "WHERE a.attempt_kind='workflow_phase' AND (p.id IS NULL OR p.command_id IS NULL OR p.control_revision IS NULL) "
                "UNION ALL SELECT 1 FROM workflow_phase_attempts p LEFT JOIN automatic_wake_attempts a ON a.id=p.native_attempt_id "
                "WHERE p.command_id IS NOT NULL AND (a.id IS NULL OR a.attempt_kind!='workflow_phase' OR a.run_ids_json!='[]' OR a.message_ids_json!='[]') LIMIT 1"
            ).fetchone()
            if invalid:
                return ("invalid_workflow_admission",)
            invalid = connection.execute(
                "SELECT 1 FROM workflow_task_claims c "
                "JOIN workflow_task_attempts t ON t.id=c.task_attempt_id "
                "JOIN workflow_phase_attempts p ON p.id=c.phase_attempt_id "
                "JOIN workflow_runs w ON w.id=p.workflow_id "
                "LEFT JOIN automatic_wake_attempts a ON a.id=p.native_attempt_id "
                "LEFT JOIN automatic_work_reservations r ON r.id=c.child_reservation_id "
                "LEFT JOIN workflow_payload_reservations result ON result.attempt_id=t.id "
                "WHERE p.purpose!='execution' OR a.attempt_kind IS NOT 'workflow_phase' "
                "OR t.workflow_id!=p.workflow_id OR t.plan_revision!=p.plan_revision OR t.phase_id!=p.phase_id "
                "OR (t.run_id IS NOT NULL AND t.run_id!=c.launch_run_id) "
                "OR result.attempt_id IS NULL OR result.workflow_id!=w.id "
                "OR (c.child_reservation_id IS NOT NULL AND (r.id IS NULL OR r.chain_id!=w.chain_id OR r.owner_id!=a.owner_id OR r.kind!='child_launch' OR r.amount!=1)) LIMIT 1"
            ).fetchone()
            if invalid:
                return ("invalid_workflow_task_claim",)
            for interval in connection.execute(
                "SELECT started_monotonic,observed_monotonic FROM workflow_active_intervals"
            ):
                if any(
                    value is not None and not math.isfinite(value) for value in interval
                ):
                    return ("invalid_workflow_active_interval",)
        for row in connection.execute("SELECT * FROM workflow_runs"):
            request = WorkflowRequest.model_validate_json(row["request_json"])
            chain = connection.execute(
                "SELECT * FROM automatic_work_chains WHERE id=?", (row["chain_id"],)
            ).fetchone()
            if (
                not chain
                or chain["conversation_id"] != request.conversation_id
                or row["conversation_id"] != request.conversation_id
                or chain["allowance_root_chain_id"] is not None
                or chain["root_submission_id"] != "workflow:" + row["id"]
            ):
                return ("invalid_workflow_ownership",)
            if hashlib.sha256(row["request_json"].encode()).hexdigest() != row[
                "payload_hash"
            ] or json.loads(chain["limits_json"]) != asdict(request.policy):
                return ("invalid_workflow_payload",)
            try:
                WorkflowRunsStore._required_results(
                    connection,
                    row["id"],
                    status=row["status"],
                    chain_id=row["chain_id"],
                )
            except ValueError:
                return ("invalid_workflow_required_result",)
            WorkflowRunsStore._capacity(connection, row["id"], 0)
            plans = list(
                connection.execute(
                    "SELECT * FROM workflow_plan_revisions WHERE workflow_id=? ORDER BY revision",
                    (row["id"],),
                )
            )
            for revision, saved in enumerate(plans, 1):
                plan = WorkflowPlan.model_validate_json(saved["payload_json"])
                if (
                    saved["revision"] != revision
                    or plan.request != request
                    or hashlib.sha256(saved["payload_json"].encode()).hexdigest()
                    != saved["payload_hash"]
                ):
                    return ("invalid_workflow_plan",)
                WorkflowRunsStore._plan_rows(connection, row["id"], revision, plan)
        for row in connection.execute("SELECT * FROM workflow_task_attempts"):
            WorkflowRunsStore._ownership(
                connection, WorkflowTaskAttempt(**dict(row)), allow_removed=True
            )
        for row in connection.execute(
            "SELECT p.*,w.conversation_id,w.chain_id FROM workflow_phase_attempts p JOIN workflow_runs w ON w.id=p.workflow_id WHERE p.native_attempt_id IS NOT NULL"
        ):
            native = connection.execute(
                "SELECT conversation_id,chain_id FROM automatic_wake_attempts WHERE id=?",
                (row["native_attempt_id"],),
            ).fetchone()
            if not native or tuple(native) != (row["conversation_id"], row["chain_id"]):
                return ("invalid_workflow_ownership",)
        for row in connection.execute("SELECT * FROM workflow_decisions"):
            decision = WorkflowDecision.model_validate_json(row["payload_json"])
            if (
                decision.id,
                decision.workflow_id,
                decision.plan_revision,
                decision.phase_id,
                decision.control_revision,
            ) != (
                row["id"],
                row["workflow_id"],
                row["plan_revision"],
                row["phase_id"],
                row["control_revision"],
            ):
                return ("invalid_workflow_ownership",)
            for ref in decision.result_refs:
                if (
                    connection.execute(
                        "SELECT 1 FROM workflow_payload_reservations WHERE attempt_id=? AND workflow_id=? AND state IN ('published','removed')",
                        (ref, decision.workflow_id),
                    ).fetchone()
                    is None
                ):
                    return ("invalid_workflow_ownership",)
        invalid = connection.execute(
            "SELECT 1 FROM workflow_payload_reservations p LEFT JOIN workflow_results r ON r.attempt_id=p.attempt_id WHERE (p.state='published' AND r.attempt_id IS NULL) OR (p.state!='published' AND r.attempt_id IS NOT NULL) OR (r.workflow_id IS NOT NULL AND r.workflow_id!=p.workflow_id) LIMIT 1"
        ).fetchone()
        if invalid:
            return ("invalid_workflow_required_result",)
        for row in connection.execute(
            "SELECT r.*,a.run_id,a.status AS attempt_status,p.bytes FROM workflow_results r JOIN workflow_task_attempts a ON a.id=r.attempt_id JOIN workflow_payload_reservations p ON p.attempt_id=r.attempt_id"
        ):
            result = WorkflowTaskResult.model_validate_json(row["payload_json"])
            if (
                (result.workflow_id, result.attempt_id, result.run_id)
                != (row["workflow_id"], row["attempt_id"], row["run_id"])
                or hashlib.sha256(row["payload_json"].encode()).hexdigest()
                != row["payload_hash"]
                or len(row["payload_json"].encode()) > row["bytes"]
                or row["attempt_status"] not in {result.status, "closed"}
            ):
                return ("invalid_workflow_result_ownership",)
        for row in connection.execute(
            "SELECT started_at,ended_at,charged_seconds FROM workflow_active_intervals"
        ):
            if any(value is not None and not math.isfinite(value) for value in row):
                return ("invalid_workflow_active_interval",)
    except (ValueError, TypeError, KeyError):
        return ("invalid_workflow_payload",)
    finally:
        connection.row_factory = previous
    return ()


# The v25 catalog above remains frozen for existing backup recognition.
WORKFLOW_PHASE_V26 = (
    next(
        sql
        for sql in WORKFLOW_OBJECTS
        if sql.startswith("CREATE TABLE workflow_phase_attempts ")
    )
    .replace(
        "workflow_id TEXT NOT NULL,",
        "workflow_id TEXT NOT NULL REFERENCES workflow_runs(id),",
        1,
    )
    .replace("plan_revision INTEGER NOT NULL,", "plan_revision INTEGER,")
    .replace("phase_id TEXT NOT NULL,", "phase_id TEXT,")
    .replace(
        "    UNIQUE(workflow_id, plan_revision, phase_id, ordinal),",
        """    purpose TEXT NOT NULL DEFAULT 'execution' CHECK(purpose IN ('planning','execution')),
    command_id TEXT UNIQUE,
    control_revision INTEGER CHECK(control_revision>0),
    result_bytes INTEGER NOT NULL DEFAULT 0 CHECK(result_bytes BETWEEN 0 AND 131072),
    CHECK((purpose='execution' AND plan_revision IS NOT NULL AND phase_id IS NOT NULL)
       OR (purpose='planning' AND phase_id IS NULL)),
    UNIQUE(workflow_id, plan_revision, phase_id, ordinal),""",
    )
)
WORKFLOW_TASK_CLAIMS_V26 = """CREATE TABLE workflow_task_claims (
    task_attempt_id TEXT PRIMARY KEY REFERENCES workflow_task_attempts(id),
    phase_attempt_id TEXT NOT NULL REFERENCES workflow_phase_attempts(id),
    child_reservation_id TEXT UNIQUE REFERENCES automatic_work_reservations(id),
    launch_run_id TEXT NOT NULL UNIQUE
)"""
WORKFLOW_INTERVAL_COLUMNS = (
    "clock_owner_id TEXT",
    "started_monotonic REAL",
    "observed_monotonic REAL",
)
WORKFLOW_ADMISSION_INDEXES = (
    "CREATE UNIQUE INDEX idx_workflow_open_interval ON workflow_active_intervals(workflow_id) WHERE ended_at IS NULL",
    "CREATE INDEX idx_workflow_task_phase ON workflow_task_claims(phase_attempt_id)",
)
WORKFLOW_ADMISSION_MIGRATION = (
    "ALTER TABLE automatic_wake_attempts RENAME COLUMN attempt_kind TO old_attempt_kind",
    "ALTER TABLE automatic_wake_attempts ADD COLUMN attempt_kind TEXT NOT NULL DEFAULT 'fleet_wake' CHECK (attempt_kind IN ('fleet_wake', 'goal_iteration', 'workflow_phase'))",
    "UPDATE automatic_wake_attempts SET attempt_kind=old_attempt_kind",
    "ALTER TABLE automatic_wake_attempts DROP COLUMN old_attempt_kind",
    "ALTER TABLE workflow_phase_attempts RENAME TO workflow_phase_attempts_v25",
    WORKFLOW_PHASE_V26,
    "INSERT INTO workflow_phase_attempts (id,workflow_id,plan_revision,phase_id,ordinal,native_attempt_id,status,retirement) SELECT id,workflow_id,plan_revision,phase_id,ordinal,native_attempt_id,status,retirement FROM workflow_phase_attempts_v25",
    "DROP TABLE workflow_phase_attempts_v25",
    WORKFLOW_TASK_CLAIMS_V26,
    *(
        "ALTER TABLE workflow_active_intervals ADD COLUMN " + column
        for column in WORKFLOW_INTERVAL_COLUMNS
    ),
    *WORKFLOW_ADMISSION_INDEXES,
    "INSERT INTO schema_version(version) VALUES (26)",
)


def workflow_admission_catalog(catalog):
    """Derive the exact fixed migration's DDL, including SQLite ALTER quoting."""
    with sqlite3.connect(":memory:") as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        for sql in WORKFLOW_ADMISSION_MIGRATION:
            conn.execute(sql)
        return tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )


WORKFLOW_CHECKPOINT_OBJECTS = (
    "CREATE TABLE workflow_checkpoints (attempt_id TEXT PRIMARY KEY REFERENCES workflow_results(attempt_id), workflow_id TEXT NOT NULL REFERENCES workflow_runs(id), payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL CHECK(length(CAST(payload_json AS BLOB)) <= 131072))",
    "CREATE TRIGGER workflow_checkpoint_immutable BEFORE UPDATE ON workflow_checkpoints BEGIN SELECT RAISE(ABORT, 'workflow checkpoint is immutable'); END",
    "CREATE TABLE workflow_resume_commands (id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL REFERENCES workflow_runs(id), control_revision INTEGER NOT NULL CHECK(control_revision>0), resulting_revision INTEGER NOT NULL CHECK(resulting_revision>control_revision))",
)
WORKFLOW_CHECKPOINT_MIGRATION = (
    *WORKFLOW_CHECKPOINT_OBJECTS,
    "INSERT INTO schema_version(version) VALUES (27)",
)


def workflow_checkpoint_catalog(catalog):
    with sqlite3.connect(":memory:") as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        for sql in WORKFLOW_CHECKPOINT_MIGRATION:
            conn.execute(sql)
        return tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
