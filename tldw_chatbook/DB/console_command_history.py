"""Passive command facts in Runs26 receipts; never command execution authority."""

import json
import re
from dataclasses import dataclass, field

from tldw_chatbook.Canvas.limits import validate_opaque_identifier
from tldw_chatbook.Chat.console_remote_models import ConsoleTarget
from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)

from .console_command_receipts import (
    ConsoleCommandReceipt,
    ConsoleCommandReceiptRepository,
    _now,
)

_MANIFEST = "history.command"
_PREFIX = _MANIFEST + ":"
_MAX_BYTES = 16 * 1024**2
_COLUMNS = (
    "command_id",
    "owner_epoch",
    "console_id",
    "action",
    "target_json",
    "payload_fingerprint",
    "state",
    "issued_ns",
    "expires_ns",
    "native_id",
    "reason",
    "updated_at",
)


@dataclass(frozen=True)
class CommandHistoryRecord:
    """Original SQL facts and separately proved Run associations, never a ticket."""

    original_fields: tuple = field(repr=False)
    run_associations: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CommandHistoryCursor:
    operation_id: str
    source_fingerprint: str
    after: str


@dataclass(frozen=True)
class CommandHistoryPage:
    items: tuple[CommandHistoryRecord, ...]
    next_cursor: CommandHistoryCursor | None


def _identity(operation_id, source_fingerprint):
    if type(operation_id) is not str or len(operation_id) > 256:
        raise ValueError("command_history_operation_invalid")
    validate_opaque_identifier(operation_id)
    if (
        type(source_fingerprint) is not str
        or re.fullmatch("[0-9a-f]{64}", source_fingerprint) is None
    ):
        raise ValueError("command_history_source_invalid")


def _native(database):
    from .AgentRuns_DB import AgentRunsDB

    if type(database) is not AgentRunsDB:
        raise ValueError("command_history_native_owner_required")


def _page_header(budget, operation_id, source_fingerprint):
    budget.container(4)
    budget.value((operation_id, source_fingerprint, _MANIFEST, _PREFIX))
    budget.container(100)  # Retained page list.
    budget.container(100)  # Immutable result tuple.
    budget.container(12)  # SQL metadata, page and continuation fields.


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("command_history_duplicate_field")
        result[key] = value
    return result


def _parse(raw, budget):
    budget.json(raw)
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise ValueError("command_history_invalid") from None


def _decode_child(payload, command_id, budget):
    if type(payload) is dict:
        budget.container(len(payload) * 2)
    if (
        type(payload) is not dict
        or set(payload) != {"version", "record", "run_associations"}
        or type(payload["version"]) is not int
        or payload["version"] != 1
    ):
        raise ValueError("command_history_invalid")
    record, pairs = payload["record"], payload["run_associations"]
    if type(record) is dict:
        budget.container(len(record) * 2 + len(_COLUMNS))
    if (
        type(record) is not dict
        or set(record) != set(_COLUMNS)
        or record["command_id"] != command_id
    ):
        raise ValueError("command_history_invalid")
    for name in _COLUMNS:
        value = record[name]
        if name in ("issued_ns", "expires_ns"):
            valid = type(value) is int
        else:
            valid = type(value) is str or (
                name in ("native_id", "reason") and value is None
            )
        if not valid:
            raise ValueError("command_history_invalid")
    budget.value(record)  # Native dict/pop and constructor keyword materialization.
    budget.json(record["owner_epoch"])
    budget.json(record["target_json"])
    # Native target/model and receipt constructor keyword/result dictionaries.
    for count in (
        len(ConsoleTarget.model_fields),
        len(ConsoleCommandReceipt.__dataclass_fields__),
    ):
        for _ in range(3):
            budget.container(count)
    try:
        decoded = ConsoleCommandReceiptRepository._decode(record)
        original_run = getattr(decoded.target, "run_id", None)
        if type(pairs) is not list or len(pairs) != (
            1 if original_run is not None else 0
        ):
            raise ValueError("command_history_association_invalid")
        if pairs:
            if (
                type(pairs[0]) is not list
                or len(pairs[0]) != 2
                or pairs[0][0] != original_run
            ):
                raise ValueError("command_history_association_invalid")
            validate_opaque_identifier(pairs[0][1])
    except (ValueError, TypeError, KeyError, RecursionError):
        raise ValueError("command_history_invalid") from None
    budget.container(len(_COLUMNS) * 3 + len(pairs) * 3 + 2)
    return CommandHistoryRecord(
        tuple((name, record[name]) for name in _COLUMNS),
        tuple(tuple(pair) for pair in pairs),
    )


def _encoded(value, budget):
    from tldw_chatbook.Backup_Recovery.mobile_runs_goal_budget import _serialization

    _serialization(value, budget)
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    if len(raw) > _MAX_BYTES:  # ensure_ascii makes character count equal byte count.
        raise ValueError("command_history_record_unreadable")
    return raw


def _prepare_command_history(operation_id, source_fingerprint, rows, budget):
    """Pre-admit all passive record encodings before the operation's first effect."""
    from tldw_chatbook.Backup_Recovery.mobile_runs_preflight import _OperationalHistory

    _identity(operation_id, source_fingerprint)
    if type(budget) is not _TraceReadBudget or type(rows) is not tuple:
        raise ValueError("command_history_native_input_required")
    budget.container(len(rows) * 3)
    prepared, seen = [], set()
    budget.container(3)
    manifest = {"version": 1, "kind": "command_history", "record_count": len(rows)}
    master = _encoded(manifest, budget)
    # Account every future complete-record read on the original allowance.
    # No record receives a fresh allowance to make an unreadable record fit.
    for row in rows:
        if (
            type(row) is not _OperationalHistory
            or type(row.values) is not tuple
            or len(row.values) != len(_COLUMNS)
        ):
            raise ValueError("command_history_native_input_required")
        budget.container(len(_COLUMNS))
        if any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[1]) not in (str, int, type(None))
            for pair in row.values
        ):
            raise ValueError("command_history_native_input_required")
        if (
            type(row) is not _OperationalHistory
            or row.table != "console_command_receipts"
            or type(row.original) is not tuple
            or len(row.original) != 1
            or type(row.associations) is not tuple
            or tuple(key for key, _ in row.values) != _COLUMNS
        ):
            raise ValueError("command_history_native_input_required")
        command_id = row.original[0]
        validate_opaque_identifier(command_id)
        if command_id in seen:
            raise ValueError("command_history_duplicate_record")
        budget.value(row.values)
        budget.value(row.associations)
        budget.value(command_id)
        budget.container(6 + 3 * len(row.associations))
        if any(
            type(pair) is not tuple or len(pair) != 3 or pair[0] != "run"
            for pair in row.associations
        ):
            raise ValueError("command_history_association_invalid")
        payload = {
            "version": 1,
            "record": dict(row.values),
            "run_associations": [[pair[1], pair[2]] for pair in row.associations],
        }
        raw = _encoded(payload, budget)
        _page_header(budget, operation_id, source_fingerprint)
        budget.reserve(64 + 4 * len(master))
        _parse(master, budget)
        budget.value(_PREFIX)
        budget.value(command_id)
        budget.reserve(64 + 4 * len(raw))
        _decode_child(_parse(raw, budget), command_id, budget)
        seen.add(command_id)
        prepared.append((_PREFIX + command_id, raw))
    budget.container(6)
    prepared.append((_MANIFEST, master))
    budget.container(len(prepared))
    return tuple(prepared)


def _commit_command_history(
    database, connection, operation_id, source_fingerprint, prepared
):
    """Write prevalidated bytes only inside the actual native owner's transaction."""
    _native(database)
    _identity(operation_id, source_fingerprint)
    database._require_execution_owner()
    with database.connection() as owned:
        if owned is not connection or not connection.in_transaction:
            raise ValueError("command_history_owner_transaction_required")
        if connection.execute(
            "SELECT 1 FROM mobile_consolidation_batches WHERE operation_id=? AND substr(batch_key,1,?)=? LIMIT 1",
            (operation_id, len(_MANIFEST), _MANIFEST),
        ).fetchone():
            raise ValueError("command_history_already_retained")
        stamp = _now()
        connection.execute("SAVEPOINT command_history_retention")
        try:
            for key, raw in prepared:
                connection.execute(
                    "INSERT INTO mobile_consolidation_batches(operation_id,source_fingerprint,batch_key,mapping_json,committed_at) VALUES (?,?,?,?,?)",
                    (operation_id, source_fingerprint, key, raw, stamp),
                )
        except BaseException:
            connection.execute("ROLLBACK TO command_history_retention")
            raise
        finally:
            connection.execute("RELEASE command_history_retention")


def _retain_command_history(
    database, connection, operation_id, source_fingerprint, rows, *, _read_budget
):
    """Retain the existing passive component without conferring import readiness."""
    _native(database)
    _identity(operation_id, source_fingerprint)
    database._require_execution_owner()
    with database.connection() as owned:
        if owned is not connection or not connection.in_transaction:
            raise ValueError("command_history_owner_transaction_required")
        prepared = _prepare_command_history(
            operation_id, source_fingerprint, rows, _read_budget
        )
        _commit_command_history(
            database, connection, operation_id, source_fingerprint, prepared
        )


def _read_payload(connection, operation_id, source_fingerprint, key, size, budget):
    if type(size) is not int or not 0 < size <= _MAX_BYTES:
        raise ValueError("command_history_record_unreadable")
    budget.reserve(64 + 4 * size)
    row = connection.execute(
        "SELECT mapping_json FROM mobile_consolidation_batches WHERE operation_id=? AND source_fingerprint=? AND batch_key=?",
        (operation_id, source_fingerprint, key),
    ).fetchone()
    if row is None or type(row[0]) is not str:
        raise ValueError("command_history_invalid")
    return _parse(row[0], budget)


def _list_command_history(
    database, operation_id, source_fingerprint, *, cursor=None, limit=50
):
    _native(database)
    _identity(operation_id, source_fingerprint)
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("command_history_page_invalid")
    if cursor is not None:
        if type(cursor) is not CommandHistoryCursor or (
            cursor.operation_id,
            cursor.source_fingerprint,
        ) != (operation_id, source_fingerprint):
            raise ValueError("command_history_cursor_invalid")
        if type(cursor.after) is not str or len(cursor.after) > 256:
            raise ValueError("command_history_cursor_invalid")
        validate_opaque_identifier(cursor.after)
    budget = _TraceReadBudget()
    _page_header(budget, operation_id, source_fingerprint)
    budget.value(cursor)
    with database.connection() as connection:
        if connection.in_transaction:
            raise ValueError("command_history_uncommitted")
        connection.execute("BEGIN DEFERRED")
        try:
            return _read_page(
                connection, operation_id, source_fingerprint, cursor, limit, budget
            )
        finally:
            connection.rollback()


def _validate_complete_history(
    connection, operation_id, source_fingerprint, expected_count, budget
):
    """Validate all passive facts in the owning caller's current snapshot/allowance."""
    if not connection.in_transaction or type(budget) is not _TraceReadBudget:
        raise ValueError("command_history_owner_snapshot_required")
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("command_history_count_invalid")
    _identity(operation_id, source_fingerprint)
    cursor, count = None, 0
    while True:
        _page_header(budget, operation_id, source_fingerprint)
        budget.value(cursor)
        page = _read_page(
            connection, operation_id, source_fingerprint, cursor, 100, budget
        )
        count += len(page.items)
        if count > expected_count:
            raise ValueError("command_history_owner_count_changed")
        following = page.next_cursor
        if following is None:
            if count != expected_count:
                raise ValueError("command_history_owner_count_changed")
            return
        if not page.items or (cursor is not None and following.after <= cursor.after):
            raise ValueError("command_history_incomplete")
        cursor = following


def _read_page(connection, operation_id, source_fingerprint, cursor, limit, budget):
    master = connection.execute(
        "SELECT length(CAST(mapping_json AS BLOB)), source_fingerprint=? FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
        (source_fingerprint, operation_id, _MANIFEST),
    ).fetchone()
    if master is None or master[1] != 1:
        raise ValueError("command_history_manifest_required")
    manifest = _read_payload(
        connection, operation_id, source_fingerprint, _MANIFEST, master[0], budget
    )
    if type(manifest) is dict:
        budget.container(len(manifest) * 2)
    if (
        type(manifest) is not dict
        or set(manifest) != {"version", "kind", "record_count"}
        or type(manifest["version"]) is not int
        or manifest["version"] != 1
        or manifest["kind"] != "command_history"
        or type(manifest["record_count"]) is not int
        or manifest["record_count"] < 0
    ):
        raise ValueError("command_history_manifest_invalid")
    census = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(source_fingerprint<>? OR substr(batch_key,1,?)<>? OR length(CAST(batch_key AS BLOB))>? OR length(CAST(batch_key AS BLOB))<=? OR typeof(mapping_json)<>'text'),0) FROM mobile_consolidation_batches WHERE operation_id=? AND substr(batch_key,1,?)=? AND batch_key<>?",
        (
            source_fingerprint,
            len(_PREFIX),
            _PREFIX,
            len(_PREFIX) + 256,
            len(_PREFIX),
            operation_id,
            len(_MANIFEST),
            _MANIFEST,
            _MANIFEST,
        ),
    ).fetchone()
    if census[0] != manifest["record_count"] or census[1]:
        raise ValueError("command_history_incomplete")
    budget.reserve(64 + 4 * (len(_PREFIX) + 256))
    if (
        cursor is not None
        and connection.execute(
            "SELECT 1 FROM mobile_consolidation_batches WHERE operation_id=? AND source_fingerprint=? AND batch_key=?",
            (operation_id, source_fingerprint, _PREFIX + cursor.after),
        ).fetchone()
        is None
    ):
        raise ValueError("command_history_cursor_invalid")
    budget.reserve(64 + 4 * (len(_PREFIX) + 256))
    after = _PREFIX + cursor.after if cursor else _PREFIX
    items = []
    last = None
    for _ in range(limit + 1):
        try:
            # Read bounded metadata only; body byte length precedes text allocation.
            budget.reserve(64 + 4 * (len(_PREFIX) + 256))
            budget.container(8)
            row = connection.execute(
                "SELECT batch_key, length(CAST(mapping_json AS BLOB)) FROM mobile_consolidation_batches WHERE operation_id=? AND source_fingerprint=? AND batch_key>? AND substr(batch_key,1,?)=? ORDER BY batch_key LIMIT 1",
                (operation_id, source_fingerprint, after, len(_PREFIX), _PREFIX),
            ).fetchone()
            if row is None:
                return CommandHistoryPage(tuple(items), None)
            if len(items) == limit:
                break
            command_id = row[0][len(_PREFIX) :]
            validate_opaque_identifier(command_id)
            payload = _read_payload(
                connection, operation_id, source_fingerprint, row[0], row[1], budget
            )
            item = _decode_child(payload, command_id, budget)
        except _TraceReadLimit:
            if not items:
                raise ValueError("command_history_record_unreadable") from None
            break
        items.append(item)
        last, after = command_id, row[0]
    return CommandHistoryPage(
        tuple(items), CommandHistoryCursor(operation_id, source_fingerprint, last)
    )
