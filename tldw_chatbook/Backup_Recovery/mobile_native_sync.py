"""Fixed native Sync effects, borrowed from the actual Chat owning transaction."""

import json

from .mobile_sync_plan import _PAYLOAD_COLUMNS

# JSON key order is not a native payload difference. Keep every scalar's JSON
# type and path; duplicate members cannot disappear behind set comparison.
_MATCH = """
SELECT count(*) FROM sync_log AS s
WHERE entity=? AND entity_id=? AND version=? AND operation=? AND timestamp=?
  AND client_id=?
  AND (SELECT count(*) FROM json_tree(s.payload))=(SELECT count(*) FROM json_tree(?))
  AND NOT EXISTS (
    SELECT fullkey,type,atom FROM json_tree(s.payload)
    EXCEPT SELECT fullkey,type,atom FROM json_tree(?)
  )
  AND NOT EXISTS (
    SELECT fullkey,type,atom FROM json_tree(?)
    EXCEPT SELECT fullkey,type,atom FROM json_tree(s.payload)
  )
"""
_COUNT = "SELECT count(*) FROM sync_log WHERE entity=? AND entity_id=?"
_INSERT = "INSERT INTO sync_log(entity,entity_id,version,operation,timestamp,client_id,payload) VALUES (?,?,?,?,?,?,?)"


def _prepare_sync(plan, budget):
    """Pre-admit native generated proof comparisons before any owner effect."""
    budget.container(len(plan.rows) * 3 + len(plan.sync_frontiers) * 10)
    rows = {(row.table, row.original): row for row in plan.rows}
    prepared = []
    seen = set()
    for frontier in plan.sync_frontiers:
        budget.container(24)
        identity = (frontier.entity, str(frontier.mapped_id))
        if identity in seen or frontier.entity not in _PAYLOAD_COLUMNS:
            raise ValueError("mobile_native_sync_frontier_ambiguous")
        seen.add(identity)
        final = (
            *identity,
            frontier.version,
            frontier.operation,
            frontier.timestamp,
            frontier.client_id,
            frontier.payload,
        )
        current = None
        if frontier.current_disposition != "retain_native_body_free_hard_delete":
            row = rows.get(frontier.owner)
            if row is None or row.table != frontier.entity:
                raise ValueError("mobile_native_sync_owner_missing")
            budget.container(len(row.values) * 2)
            values = dict(row.values)
            budget.container(len(_PAYLOAD_COLUMNS[frontier.entity]) * 2)
            payload = {
                column: values[column]
                for column in _PAYLOAD_COLUMNS[frontier.entity].split()
            }
            budget.value(payload, text_copies=3)
            raw = json.dumps(
                payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            current = (
                *identity,
                frontier.version,
                "create",
                frontier.timestamp,
                frontier.client_id,
                raw,
            )
            if frontier.current_disposition not in {
                "native_insert_current_unique",
                "replace_native_create_with_body_free_tombstone",
            }:
                raise ValueError("mobile_native_sync_disposition_invalid")
            if frontier.current_disposition == "native_insert_current_unique":
                budget.json(frontier.payload)
                if json.loads(frontier.payload) != payload:
                    raise ValueError("mobile_native_sync_generated_payload_changed")
        elif frontier.entity != "world_book_entries" or frontier.operation != "delete":
            raise ValueError("mobile_native_sync_hard_delete_invalid")
        previous = None
        if frontier.previous_payload is not None:
            if frontier.entity != "messages" or frontier.version <= 1:
                raise ValueError("mobile_native_sync_previous_invalid")
            previous = (
                *identity,
                frontier.version - 1,
                frontier.previous_operation,
                frontier.previous_timestamp,
                frontier.previous_client_id,
                frontier.previous_payload,
            )
        # Fixed scalar result/parameter tuples for insertion, both validations
        # and receipt-first resume are reserved before the first native effect.
        budget.container(120)
        prepared.append((frontier, current, final, previous))
    budget.container(len(prepared))
    return tuple(prepared)


def _require_transaction(connection):
    native = getattr(connection, "connection", connection)
    if not native.in_transaction:
        raise ValueError("mobile_sync_owner_transaction_required")


def _match(connection, parameters):
    raw = parameters[-1]
    return (
        connection.execute(_MATCH, (*parameters[:-1], raw, raw, raw)).fetchone()[0] == 1
    )


def _apply_sync(connection, prepared):
    """Verify the actual INSERT trigger before retaining a prior proof/tombstone."""
    _require_transaction(connection)
    for frontier, current, final, previous in prepared:
        count = connection.execute(_COUNT, final[:2]).fetchone()[0]
        if current is None:
            if count != 0:
                raise ValueError("mobile_native_sync_hard_delete_collision")
        elif count != 1 or not _match(connection, current):
            raise ValueError("mobile_native_sync_current_changed")
        if frontier.current_disposition != "native_insert_current_unique":
            if current is not None:
                connection.execute(
                    "DELETE FROM sync_log WHERE entity=? AND entity_id=? AND version=? AND operation='create'",
                    current[:3],
                )
            connection.execute(_INSERT, final)
        if previous is not None:
            connection.execute(_INSERT, previous)
    _verify_sync(connection, prepared)


def _verify_sync(connection, prepared):
    """Receipt resume checks exact current/previous proofs without replaying INSERT."""
    _require_transaction(connection)
    for _frontier, _current, final, previous in prepared:
        if (
            connection.execute(_COUNT, final[:2]).fetchone()[0]
            != 1 + (previous is not None)
            or not _match(connection, final)
            or (previous is not None and not _match(connection, previous))
        ):
            raise ValueError("mobile_native_sync_frontier_changed")
