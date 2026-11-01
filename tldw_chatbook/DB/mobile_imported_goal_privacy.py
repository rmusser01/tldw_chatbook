"""Transaction-local historical accounting proof for explicit Goal body erasure."""

from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

from .mobile_consolidation_receipts import _SPECS, _read_owner_receipt, _row_digest

# Fixed native relationships; these selectors do not grant execution authority.
_FAMILY = """WITH family AS (
 SELECT id,conversation_id FROM automatic_work_chains
 WHERE id=? OR allowance_root_chain_id=?
), goals AS (
 SELECT id FROM goal_runs WHERE chain_id IN (SELECT id FROM family)
 OR conversation_id IN (SELECT conversation_id FROM family)
), wakes AS (
 SELECT id FROM automatic_wake_attempts WHERE chain_id IN (SELECT id FROM family)
 OR conversation_id IN (SELECT conversation_id FROM family)
), runs AS (
 SELECT id FROM agent_runs WHERE work_chain_id IN (SELECT id FROM family)
 OR conversation_id IN (SELECT conversation_id FROM family)
 OR id IN (SELECT run_id FROM goal_iterations WHERE goal_id IN (SELECT id FROM goals))
 OR id IN (SELECT run_id FROM automatic_wake_claims WHERE attempt_id IN (SELECT id FROM wakes))
 OR id IN (SELECT source_run_id FROM automatic_progress_wake_claims WHERE attempt_id IN (SELECT id FROM wakes))
 OR id IN (SELECT source_run_id FROM automatic_chat_start_attempts WHERE chain_id IN (SELECT id FROM family) OR source_chain_id IN (SELECT id FROM family))
) """
_SELECTORS = {
    "automatic_work_chains": "id IN (SELECT id FROM family)",
    "agent_runs": "id IN (SELECT id FROM runs)",
    "automatic_work_reservations": "chain_id IN (SELECT id FROM family)",
    "automatic_wake_attempts": "id IN (SELECT id FROM wakes)",
    "automatic_chat_start_attempts": "chain_id IN (SELECT id FROM family) OR source_chain_id IN (SELECT id FROM family) OR source_run_id IN (SELECT id FROM runs)",
    "automatic_wake_claims": "attempt_id IN (SELECT id FROM wakes) OR run_id IN (SELECT id FROM runs)",
    "automatic_progress_wake_claims": "attempt_id IN (SELECT id FROM wakes) OR source_run_id IN (SELECT id FROM runs)",
    "goal_runs": "id IN (SELECT id FROM goals)",
    "goal_iterations": "goal_id IN (SELECT id FROM goals)",
    "goal_checkpoints": "goal_id IN (SELECT id FROM goals)",
    "goal_evidence": "goal_id IN (SELECT id FROM goals)",
    "goal_reports": "goal_id IN (SELECT id FROM goals)",
    "goal_payload_reservations": "goal_id IN (SELECT id FROM goals)",
    "goal_waits": "goal_id IN (SELECT id FROM goals)",
}
_IDENTITIES = {
    "agent_runs": "run",
    "automatic_work_chains": "work_chain",
    "automatic_work_reservations": "work_reservation",
    "automatic_wake_attempts": "wake_attempt",
    "automatic_chat_start_attempts": "chat_start_attempt",
    "goal_runs": "goal",
    "goal_iterations": "goal_iteration",
    "goal_evidence": "goal_evidence",
    "goal_reports": "goal_report",
    "goal_checkpoints": "wake_attempt",
    "goal_payload_reservations": "wake_attempt",
    "goal_waits": "goal",
    "automatic_wake_claims": "run",
}
_HOLDS = {
    "agent_runs": "run",
    "automatic_work_chains": "chain",
    "automatic_wake_attempts": "wake_attempt",
    "automatic_chat_start_attempts": "chat_start_attempt",
    "automatic_progress_wake_claims": "progress",
    "goal_runs": "goal",
}


def _refuse():
    raise ValueError("goal_not_settled")


def _rows(connection, table, predicate, params, budget, *, prefix=""):
    """Price SQL-encoded bytes before fetching any native body or hash copy."""
    spec = _SPECS[table]
    budget.reserve(
        4096
        + 16 * (sum(map(len, spec.columns)) + len(table) + len(predicate) + len(prefix))
    )
    columns = ",".join('"' + column + '"' for column in spec.columns)
    sizes = "+".join(
        'coalesce(length(CAST("' + column + '" AS BLOB)),0)' for column in spec.columns
    )
    suffix = ' FROM "' + table + '" WHERE ' + predicate
    count, size = connection.execute(
        prefix + "SELECT count(*),coalesce(sum(" + sizes + "),0)" + suffix, params
    ).fetchone()
    budget.reserve(size * 4 + count * len(spec.columns) * 192)
    budget.container(count * 2)
    cursor = connection.execute(prefix + "SELECT " + columns + suffix, params)
    try:
        while True:
            budget.container(100)
            page = cursor.fetchmany(100)
            if not page:
                return
            yield from page
    finally:
        cursor.close()


def _goal_for_privacy(connection, goal_id, budget):
    budget.value(goal_id)
    rows = list(_rows(connection, "goal_runs", "id=?", (goal_id,), budget))
    return rows[0] if rows else None


def _historical_goal_family(db, connection, goal, budget):
    """Exclude only proved historical facts; do not settle or release anything."""
    if not connection.in_transaction or type(budget) is not _TraceReadBudget:
        _refuse()
    budget.reserve(4096)
    count, operation_size, source_size = connection.execute(
        "SELECT count(*),coalesce(max(length(CAST(operation_id AS BLOB))),0),coalesce(max(length(CAST(source_fingerprint AS BLOB))),0) FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='runs'"
    ).fetchone()
    if operation_size > 256 or source_size > 64:
        _refuse()
    budget.reserve(count * (256 * 4 + 64 * 4))
    budget.container(count * 6)
    records, identities = {}, {}
    cursor = connection.execute(
        "SELECT DISTINCT operation_id,source_fingerprint FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='runs' ORDER BY operation_id,source_fingerprint"
    )
    try:
        while True:
            budget.container(100)
            page = cursor.fetchmany(100)
            if not page:
                break
            for operation, source in page:
                receipt = _read_owner_receipt(
                    connection, operation, source, "runs", budget
                )
                if receipt is None:
                    _refuse()
                for child in receipt[1]:
                    data = child["data"]
                    if child["kind"] == "row" and data["table"] in _SELECTORS:
                        budget.container(8 + len(data["mapped"]))
                        key = data["table"], tuple(data["mapped"])
                        if key in records:
                            _refuse()
                        records[key] = operation, data
                    elif (
                        child["kind"] == "identity"
                        and data["domain"] in _IDENTITIES.values()
                    ):
                        budget.container(6)
                        key = data["domain"], data["mapped"]
                        if key in identities:
                            _refuse()
                        identities[key] = operation, data["original"]
    finally:
        cursor.close()
    imported = ("goal_runs", (goal["id"],)) in records
    held = connection.execute(
        "SELECT 1 FROM mobile_import_authority_holds WHERE authority_kind='goal' AND authority_id=?",
        (goal["id"],),
    ).fetchone()
    if not imported:
        if held:
            _refuse()
        return False
    # Price the native resolver's full row copies before invoking it. The native
    # direct-root validation still decides the actual allowance relationship.
    chain_rows = list(
        _rows(connection, "automatic_work_chains", "id=?", (goal["chain_id"],), budget)
    )
    if len(chain_rows) != 1:
        _refuse()
    budget.value(chain_rows[0])
    root_id = chain_rows[0]["allowance_root_chain_id"]
    if root_id is not None:
        root_rows = list(
            _rows(connection, "automatic_work_chains", "id=?", (root_id,), budget)
        )
        if len(root_rows) != 1:
            _refuse()
        budget.value(root_rows[0])
    root = db.automatic_work._allowance_chain(connection, goal["chain_id"])["id"]
    budget.value(root)
    # Missing accounting identities cannot be assigned to a family from a hash.
    # Reject incomplete candidates; never reinterpret their absence as settlement.
    for table, mapped in records:
        if table in {
            "goal_checkpoints",
            "goal_evidence",
            "goal_reports",
            "goal_payload_reservations",
            "goal_waits",
        }:
            # These are the native privacy deletions, never execution settlement.
            # Any surviving body in the actual family still needs its exact hash.
            continue
        budget.container(len(mapped) + 2)
        predicate = " AND ".join(
            '"' + column + '"=?' for column in _SPECS[table].identity_columns
        )
        if (
            connection.execute(
                'SELECT 1 FROM "' + table + '" WHERE ' + predicate, mapped
            ).fetchone()
            is None
        ):
            _refuse()
    seen_goal = False
    for table, predicate in _SELECTORS.items():
        for row in _rows(
            connection, table, predicate, (root, root), budget, prefix=_FAMILY
        ):
            spec = _SPECS[table]
            budget.container(len(spec.columns) * 2 + len(spec.identity_columns) + 6)
            mapped = tuple(row[column] for column in spec.identity_columns)
            proof = records.get((table, mapped))
            if proof is None:
                _refuse()
            operation, record = proof
            if _row_digest(table, dict(row), budget) != record["mapped_sha256"]:
                _refuse()
            domain = _IDENTITIES.get(table)
            if domain is not None and identities.get((domain, mapped[0])) != (
                operation,
                record["original"][0],
            ):
                _refuse()
            kind = _HOLDS.get(table)
            if (
                kind is not None
                and connection.execute(
                    "SELECT 1 FROM mobile_import_authority_holds WHERE operation_id=? AND authority_kind=? AND authority_id=? AND reviewed_at IS NULL",
                    (operation, kind, mapped[0]),
                ).fetchone()
                is None
            ):
                _refuse()
            if table == "goal_runs" and row["id"] == goal["id"]:
                seen_goal = True
    if not seen_goal:
        _refuse()
    return True
