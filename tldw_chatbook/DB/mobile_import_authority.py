"""Fixed installed migration and native imported-authority quarantine policy."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path


def _migration(name: str) -> tuple[str, ...]:
    return tuple(
        statement.strip()
        for statement in (Path(__file__).with_name("migrations") / name)
        .read_text(encoding="utf-8")
        .split(";")
        if statement.strip()
    )


CHAT_MIGRATION = _migration("chachanotes_v81_to_v82_mobile_consolidation.sql")
WORKSPACE_MIGRATION = _migration("workspaces_v8_to_v9_mobile_consolidation.sql")
AGENT_RUNS_MIGRATION = _migration("agent_runs_v25_to_v26_mobile_consolidation.sql")
AUTHORITY_KINDS = frozenset(
    {"run", "chain", "goal", "wake_attempt", "progress", "chat_start_attempt"}
)


def hold_authorities(
    conn: sqlite3.Connection,
    cursor: sqlite3.Connection | sqlite3.Cursor,
    operation_id: str,
    authorities: Iterable[tuple[str, str]],
    *,
    workspace: bool = False,
) -> None:
    """Borrow only the native owner's active transaction; never commit it."""
    if not conn.in_transaction or (
        cursor is not conn and getattr(cursor, "connection", None) is not conn
    ):
        raise ValueError("import_hold_requires_owner_transaction")
    rows = tuple(authorities)
    kinds = {"binding"} if workspace else AUTHORITY_KINDS
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError("invalid import operation")
    if any(
        kind not in kinds or not isinstance(identity, str) or not identity.strip()
        for kind, identity in rows
    ):
        raise ValueError("invalid import authority")
    for kind, identity in rows:
        existing = conn.execute(
            "SELECT operation_id FROM mobile_import_authority_holds WHERE authority_kind=? AND authority_id=?",
            (kind, identity),
        ).fetchone()
        if existing is not None and existing[0] != operation_id:
            raise ValueError("import authority already claimed")
        conn.execute(
            "INSERT OR IGNORE INTO mobile_import_authority_holds (operation_id, authority_kind, authority_id) VALUES (?, ?, ?)",
            (operation_id, kind, identity),
        )


def authority_allowed(conn: sqlite3.Connection, kind: str, identity: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM mobile_import_authority_holds WHERE authority_kind=? AND authority_id=? AND reviewed_at IS NULL",
            (kind, identity),
        ).fetchone()
        is None
    )


def chain_allowed(conn: sqlite3.Connection, chain_id: str | None) -> bool:
    """Shared allowance descendants cannot borrow held root authority."""
    if chain_id is None:
        return True
    return (
        conn.execute(
            """
        WITH family AS (
            SELECT id FROM automatic_work_chains
            WHERE COALESCE(allowance_root_chain_id, id) = (
                SELECT COALESCE(allowance_root_chain_id, id)
                FROM automatic_work_chains WHERE id=?
            )
        )
        SELECT 1 FROM mobile_import_authority_holds h
        WHERE h.reviewed_at IS NULL AND (
            (h.authority_kind='chain' AND h.authority_id IN (SELECT id FROM family))
            OR (h.authority_kind='run' AND h.authority_id IN (
                SELECT id FROM agent_runs WHERE work_chain_id IN (SELECT id FROM family)))
            OR (h.authority_kind='goal' AND h.authority_id IN (
                SELECT id FROM goal_runs WHERE chain_id IN (SELECT id FROM family)))
            OR (h.authority_kind='wake_attempt' AND h.authority_id IN (
                SELECT id FROM automatic_wake_attempts WHERE chain_id IN (SELECT id FROM family)))
            OR (h.authority_kind='chat_start_attempt' AND h.authority_id IN (
                SELECT id FROM automatic_chat_start_attempts WHERE chain_id IN (SELECT id FROM family)
                OR source_chain_id IN (SELECT id FROM family)))
            OR (h.authority_kind='progress' AND h.authority_id IN (
                SELECT p.message_id FROM automatic_progress_wake_claims p
                JOIN automatic_wake_attempts a ON a.id=p.attempt_id
                WHERE a.chain_id IN (SELECT id FROM family)))
        ) LIMIT 1
    """,
            (chain_id,),
        ).fetchone()
        is None
    )


def run_allowed(conn: sqlite3.Connection, run_id: str) -> bool:
    if not authority_allowed(conn, "run", run_id):
        return False
    row = conn.execute(
        "SELECT work_chain_id FROM agent_runs WHERE id=?", (run_id,)
    ).fetchone()
    return row is None or chain_allowed(conn, row[0])
