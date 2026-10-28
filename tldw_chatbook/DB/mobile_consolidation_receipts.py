"""Strict bounded historical owner receipts; decoding is never write authority."""

from __future__ import annotations

import hashlib
import json
import struct

from tldw_chatbook.Backup_Recovery.mobile_identity_plan import _COMPOSITES, _DOMAINS
from tldw_chatbook.Backup_Recovery.mobile_plan_manifest import _TABLES
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

_OWNERS = {
    "chat": "db.chachanotes.primary",
    "workspaces": "db.workspaces",
    "runs": "db.agent_runs",
}
_KINDS = (
    "identity",
    "row",
    "provenance",
    "runlog_root",
    "runlog",
    "history_publication",
)
_MAX_BYTES = 16 * 1024 * 1024
_SPECS = {spec.table: spec for spec in _TABLES}
_DOMAINS_BY_NAME = {
    domain.name: (domain.family, (domain.scalar,)) for domain in _DOMAINS
}
for _name, _family, _table, _columns in _COMPOSITES:
    _DOMAINS_BY_NAME[_name] = (
        _family,
        tuple(
            _DOMAINS_BY_NAME[t][1][0] if isinstance(t, str) else t for _, t in _columns
        ),
    )
_DOMAINS_BY_NAME.update(
    {
        name: ("runs", (str,))
        for name in ("goal_launch", "goal_iteration_launch", "work_submission")
    }
)
# Fixed _ordered reservations in mobile_identity_plan: both are native Chat
# sequence integers, separate from opaque fleet_message and memory identities.
_DOMAINS_BY_NAME.update(
    {"memory_sequence": ("chat", (int,)), "fleet_sequence": ("chat", (int,))}
)
_COMPOSITE_NAMES = frozenset(row[0] for row in _COMPOSITES)


def _invalid():
    raise ValueError("mobile_owner_receipt_invalid")


def _integer(value):
    return type(value) is int and -(2**63) <= value < 2**63


def _nonnegative(value):
    return _integer(value) and value >= 0


def _string(value):
    return type(value) is str


def _hash(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _optional(check):
    return lambda value: value is None or check(value)


def _array(check):
    return lambda value: type(value) is list and all(check(item) for item in value)


def _one_of(*values):
    return lambda value: type(value) is str and value in values


def _scalar_id(value):
    return _integer(value) or type(value) is str


def _fields(value, schema):
    if (
        type(value) is not dict
        or len(value) != len(schema)
        or any(key not in schema for key in value)
    ):
        _invalid()
    for key, check in schema.items():
        if not check(value[key]):
            raise ValueError("mobile_owner_receipt_invalid_field:" + key)


_ID_ARRAY = _array(_scalar_id)
_HASH_OR_NULL = _optional(_hash)
_NATIVE_HASH_OR_NULL = _optional(
    lambda v: type(v) is str and v.startswith("sha256:") and _hash(v[7:])
)
_STR_OR_NULL = _optional(_string)
_OPERATION = _one_of("create", "update", "delete")
_ENTITIES = (
    "messages",
    "conversations",
    "notes",
    "character_cards",
    "keywords",
    "keyword_collections",
    "chat_dictionaries",
    "world_books",
    "world_book_entries",
)
_TABLE = lambda value: type(value) is str and value in _SPECS
_DOMAIN = lambda value: type(value) is str and value in _DOMAINS_BY_NAME
_SEMANTIC_RULES = {
    "handoff_run": (("C2b",), ("conversations",), str),
    "run": (("A-runs",), ("conversations", "messages", "fleet_progress_messages"), str),
    "goal_launch": (("A-runs",), ("conversations",), str),
    "binding": (("I",), ("conversations",), str),
    "trace_turn": (("T",), ("messages",), str),
    "chat_start_attempt": (
        ("C2b", "A-runs"),
        ("messages", "console_dispatch_checkpoints"),
        str,
    ),
    "canvas_turn": (("C2b", "T"), ("messages",), str),
    "character": (("C2c",), ("conversations", "messages"), int),
    "character_authority": (("T",), ("conversations",), int),
    "persona": (("C2c",), ("conversations",), str),
    "discovery_character": (("C2c",), ("conversations",), int),
    "discovery_persona": (("C2c",), ("conversations",), str),
    "remote_assistant": (("C2c",), ("conversations",), str),
    "conversation_memory": (("C3",), ("conversations",), str),
    "provider_continuation": (("C2b", "S"), ("messages",), str),
    "dispatch_owner": (("C2b", "S"), ("messages",), str),
    "work_chain": (("A-runs",), ("fleet_progress_messages",), str),
    **{
        domain: (("C2c",), ("messages",), int)
        for domain in (
            "visual_pack",
            "visual_version",
            "character_expression",
            "visual_asset",
        )
    },
}
_PROVENANCE = {
    "goal": {
        "table": _one_of(
            "goal_runs",
            "goal_checkpoints",
            "goal_evidence",
            "goal_reports",
            "goal_iterations",
        ),
        "original": _ID_ARRAY,
        "original_hash": _hash,
        "mapped_hash": _hash,
        "artifact_digest": _HASH_OR_NULL,
    },
    "note_snapshot": {
        "original_note": _integer,
        "original_snapshot": _optional(_integer),
        "mapped_snapshot": _optional(_integer),
        "original_run": _string,
        "original_root": _string,
        "missing": lambda v: type(v) is bool,
    },
    "snapshot_allocator": {
        "source_sequence": _optional(_nonnegative),
        "canonical_sequence": _optional(_nonnegative),
        "required_frontier": lambda v: _nonnegative(v) and v < 2**63 - 1,
    },
    "sync_source": {
        "change_id": _integer,
        "entity": _one_of(*_ENTITIES),
        "original_id": _string,
        "operation": _OPERATION,
        "timestamp": _string,
        "client_id": _string,
        "version": lambda v: _integer(v) and v > 0,
        "source_digest": _hash,
        "source_manifest_digest": _hash,
    },
    "sync_frontier": {
        "entity": _one_of(*_ENTITIES),
        "original_id": _scalar_id,
        "mapped_id": _scalar_id,
        "version": lambda v: _integer(v) and v > 0,
        "operation": _OPERATION,
        "timestamp": _string,
        "client_id": _string,
        "payload_sha256": _hash,
        "payload_hash": _NATIVE_HASH_OR_NULL,
        "previous_payload_sha256": _HASH_OR_NULL,
        "previous_hash": _NATIVE_HASH_OR_NULL,
        "previous_operation": _optional(_OPERATION),
        "previous_timestamp": _STR_OR_NULL,
        "previous_client_id": _STR_OR_NULL,
        "current_disposition": _one_of(
            "native_insert_current_unique",
            "replace_native_create_with_body_free_tombstone",
            "retain_native_body_free_hard_delete",
        ),
        "source_change_ids": _array(_integer),
        "owner_table": _TABLE,
        "owner_original": _ID_ARRAY,
    },
    "operational": {
        "table": _TABLE,
        "original": _ID_ARRAY,
        "source_digest": _hash,
        "source_manifest_digest": _hash,
        "disposition": _string,
    },
    "runtime_owner": {
        "singleton": lambda v: type(v) is int and v == 1,
        "owner_id": _string,
    },
    "semantic": {
        "table": _TABLE,
        "original": _ID_ARRAY,
        "handler": _one_of("C2b", "C2c", "C3", "T", "S", "A-runs", "I"),
        "domain": _one_of(*_SEMANTIC_RULES),
        "original_id": _scalar_id,
        "mapped_id": _scalar_id,
    },
    "citation": {
        "original": _ID_ARRAY,
        "source_manifest_digest": _hash,
        "canonical_manifest_digest": _hash,
        "disposition": _one_of("sql_no_history_context_not_adopted"),
    },
    "trace_incidence": {
        "original_call_id": _string,
        "mapped_call_id": _string,
        "source_manifest_digest": _hash,
        "canonical_manifest_digest": _hash,
        "inputs": _array(
            lambda v: (
                type(v) is list and len(v) == 2 and _TABLE(v[0]) and _ID_ARRAY(v[1])
            )
        ),
    },
    "workflow": {
        "table": _one_of("message_trajectory_metadata", "transcript_annotations"),
        "original": _ID_ARRAY,
        "source_identity": _string,
        "projected_identity": _string,
        "meaning": _one_of(
            "historical_source_event", "native_sidecar_event", "native_message_anchor"
        ),
    },
}
_OPERATIONAL = {
    **{
        f"character_conversation_search_{suffix}": "retain_destination_authority_native_derived"
        for suffix in ("dirty", "documents", "generations", "revision", "state")
    },
    **{
        f"console_trace_{suffix}": "retain_canonical_operational_authority"
        for suffix in (
            "graph_epoch",
            "migration_state",
            "maintenance_state",
            "compaction_state",
        )
    },
}


def _relative(value):
    return (
        type(value) is str
        and bool(value)
        and not value.startswith("/")
        and "\\" not in value
        and "\0" not in value
        and all(part not in ("", ".", "..") for part in value.split("/"))
    )


def _physical_integer(value):
    return type(value) is int and 0 <= value <= 2**64 - 1


def _artifact(value):
    _fields(
        value,
        {
            "path": lambda v: _string(v) and v.startswith("/") and "\0" not in v,
            "device": _physical_integer,
            "inode": _physical_integer,
            "kind": _one_of("file", "directory"),
            "size": _nonnegative,
            "sha256": _hash,
        },
    )
    return True


def _pair(value):
    return (
        type(value) is list
        and len(value) == 2
        and all(type(item) is str and item for item in value)
    )


def _validate_child(value, owner):
    _fields(
        value,
        {
            "version": lambda v: type(v) is int and v == 1,
            "kind": _one_of(*_KINDS),
            "data": lambda v: type(v) is dict,
        },
    )
    kind, data = value["kind"], value["data"]
    if kind == "identity":
        _fields(
            data,
            {"domain": _DOMAIN, "original": lambda v: True, "mapped": lambda v: True},
        )
        family, types = _DOMAINS_BY_NAME[data["domain"]]
        if family != owner:
            _invalid()
        for key in ("original", "mapped"):
            parts = data[key] if data["domain"] in _COMPOSITE_NAMES else [data[key]]
            if (
                type(parts) is not list
                or len(parts) != len(types)
                or any(
                    type(item) is not expected
                    or (expected is int and not _integer(item))
                    for item, expected in zip(parts, types, strict=True)
                )
            ):
                _invalid()
    elif kind == "row":
        _fields(
            data,
            {
                "table": _TABLE,
                "original": _ID_ARRAY,
                "mapped": _ID_ARRAY,
                "source_sha256": _hash,
                "mapped_sha256": _hash,
            },
        )
        spec = _SPECS[data["table"]]
        if (
            spec.family != owner
            or len(data["original"]) != len(spec.identity_columns)
            or len(data["mapped"]) != len(spec.identity_columns)
        ):
            _invalid()
    elif kind == "provenance":
        _fields(
            data, {"family": _one_of(*_PROVENANCE), "record": lambda v: type(v) is dict}
        )
        family, record = data["family"], data["record"]
        _fields(record, _PROVENANCE[family])
        expected = (
            "runs"
            if family
            in {"goal", "note_snapshot", "snapshot_allocator", "runtime_owner"}
            else "chat"
        )
        if family == "semantic":
            expected = _SPECS[record["table"]].family
            handlers, tables, scalar = _SEMANTIC_RULES[record["domain"]]
            if (
                record["handler"] not in handlers
                or record["table"] not in tables
                or type(record["original_id"]) is not scalar
                or type(record["mapped_id"]) is not scalar
            ):
                _invalid()
        if expected != owner:
            _invalid()
        if (
            family == "operational"
            and _OPERATIONAL.get(record["table"]) != record["disposition"]
        ):
            _invalid()
        if family == "workflow" and (
            (record["table"] == "transcript_annotations")
            != (record["meaning"] == "native_message_anchor")
        ):
            _invalid()
    elif kind == "runlog_root":
        _fields(
            data, {"root_id": _string, "location": _relative, "artifact": _artifact}
        )
        if owner != "runs" or data["artifact"]["kind"] != "directory":
            _invalid()
    elif kind == "runlog":
        _fields(
            data,
            {
                "logical_id": _string,
                "root_id": _string,
                "relative_path": lambda v: v == "" or _relative(v),
                "location": _relative,
                "size": _nonnegative,
                "sha256": _HASH_OR_NULL,
                "owner": _optional(_pair),
                "framed_runs": _array(_pair),
                "historical_run_ids": _array(_string),
                "segment_ids": _array(_string),
                "artifact": _artifact,
            },
        )
        if owner != "runs" or (data["sha256"] is None) != (
            data["artifact"]["kind"] == "directory"
        ):
            _invalid()
        if data["relative_path"] == "":
            if (
                data["logical_id"] != data["root_id"]
                or data["sha256"] is not None
                or data["size"] != 0
                or data["owner"] is not None
                or data["framed_runs"]
                or data["historical_run_ids"]
                or data["segment_ids"]
            ):
                _invalid()
        elif data["logical_id"] == data["root_id"]:
            _invalid()
    else:
        _fields(
            data,
            {
                "destination_identity": _hash,
                "source_fingerprint": _hash,
                "location": lambda v: _string(v) and v.startswith("/"),
                "artifact": _artifact,
            },
        )
        if owner != "runs" or data["artifact"]["kind"] != "directory":
            _invalid()


def _validate_master(value, owner):
    if owner not in _OWNERS:
        _invalid()
    _fields(
        value,
        {
            "version": lambda v: type(v) is int and v == 1,
            "kind": _one_of("owner_completion"),
            "owner": _one_of(_OWNERS[owner]),
            "source_manifest_sha256": _HASH_OR_NULL,
            "canonical_manifest_sha256": _HASH_OR_NULL,
            "entry_count": _nonnegative,
            "entries_sha256": _hash,
            "command_history_count": _optional(_nonnegative),
            "source_runs_version": lambda v: (
                v is None or (type(v) is int and v in (24, 25))
            ),
        },
    )
    if owner == "runs":
        if (
            value["command_history_count"] is None
            or (
                value["source_runs_version"] is not None
                and value["source_manifest_sha256"] is None
            )
            or (
                value["source_runs_version"] in (None, 24)
                and value["command_history_count"] != 0
            )
        ):
            _invalid()
    elif (
        value["command_history_count"] is not None
        or value["source_runs_version"] is not None
    ):
        _invalid()


def _encode(value, budget):
    budget.value(value, text_copies=8)
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded = raw.encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError):
        _invalid()
    if len(encoded) > _MAX_BYTES:
        _invalid()
    return raw


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid()
        result[key] = value
    return result


def _decode(raw, budget):
    if type(raw) is not str:
        _invalid()
    budget.json(raw)
    try:
        value = json.loads(
            raw, object_pairs_hook=_unique, parse_constant=lambda _: _invalid()
        )
    except (ValueError, RecursionError):
        _invalid()
    if _encode(value, budget) != raw:
        _invalid()
    return value


def _digest_entry(digest, key, raw, budget):
    budget.reserve(4 * (len(key) + len(raw)) + 128)
    for value in (key, raw):
        data = value.encode("utf-8")
        digest.update(struct.pack(">Q", len(data)))
        digest.update(data)


def _validate_census(children, owner, budget):
    budget.container(len(children) * 6)
    identities, targets, rows, row_targets, roots, members = (
        set(),
        set(),
        set(),
        set(),
        {},
        set(),
    )
    root_members = set()
    allocator_count = publication_count = 0
    for child in children:
        kind, data = child["kind"], child["data"]
        if kind == "identity":
            budget.container(4)
            original = (
                tuple(data["original"])
                if type(data["original"]) is list
                else data["original"]
            )
            mapped = (
                tuple(data["mapped"])
                if type(data["mapped"]) is list
                else data["mapped"]
            )
            source_key, target_key = (
                (data["domain"], original),
                (data["domain"], mapped),
            )
            if source_key in identities or target_key in targets:
                _invalid()
            identities.add(source_key)
            targets.add(target_key)
        elif kind == "row":
            budget.container(len(data["original"]) + len(data["mapped"]) + 4)
            source_key, target_key = (
                (data["table"], tuple(data["original"])),
                (data["table"], tuple(data["mapped"])),
            )
            if source_key in rows or target_key in row_targets:
                _invalid()
            rows.add(source_key)
            row_targets.add(target_key)
        elif kind == "provenance" and data["family"] == "snapshot_allocator":
            allocator_count += 1
        elif kind == "runlog_root":
            if data["root_id"] in roots:
                _invalid()
            roots[data["root_id"]] = data
        elif kind == "runlog":
            if data["logical_id"] in members:
                _invalid()
            members.add(data["logical_id"])
        elif kind == "history_publication":
            publication_count += 1
    if owner == "runs" and allocator_count != 1:
        _invalid()
    if publication_count > 1 or (bool(roots or members) != bool(publication_count)):
        _invalid()
    for child in children:
        if child["kind"] != "runlog":
            continue
        data = child["data"]
        root = roots.get(data["root_id"])
        if root is None:
            _invalid()
        if data["relative_path"] == "":
            if (
                data["root_id"] in root_members
                or data["location"] != root["location"]
                or data["artifact"] != root["artifact"]
            ):
                _invalid()
            root_members.add(data["root_id"])
    if root_members != set(roots):
        _invalid()


def _encode_owner_receipt(
    owner,
    children,
    *,
    source_manifest_sha256,
    canonical_manifest_sha256,
    source_runs_version,
    command_history_count,
    budget,
):
    if owner not in _OWNERS or type(budget) is not _TraceReadBudget:
        _invalid()
    budget.container(len(children) * 5)
    for child in children:
        _validate_child(child, owner)
    _validate_census(children, owner, budget)
    ordered = []
    for child in children:
        _validate_child(child, owner)
        raw = _encode(child, budget)
        decoded = _decode(raw, budget)  # prospective complete consumer readability
        _validate_child(decoded, owner)
        data = _encode(child["data"], budget)
        budget.reserve(4 * len(data) + 64)
        ordered.append((_KINDS.index(child["kind"]), data.encode("utf-8"), raw))
    ordered.sort(key=lambda row: (row[0], row[1]))
    digest, encoded = hashlib.sha256(), []
    previous = None
    for ordinal, (kind, data, raw) in enumerate(ordered):
        if previous == (kind, data):
            _invalid()
        previous = kind, data
        key = f"{owner}.entry:{ordinal}"
        budget.value(key)
        _digest_entry(digest, key, raw, budget)
        encoded.append((key, raw))
    master = {
        "version": 1,
        "kind": "owner_completion",
        "owner": _OWNERS[owner],
        "source_manifest_sha256": source_manifest_sha256,
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "entry_count": len(encoded),
        "entries_sha256": digest.hexdigest(),
        "command_history_count": command_history_count,
        "source_runs_version": source_runs_version,
    }
    _validate_master(master, owner)
    raw = _encode(master, budget)
    _validate_master(_decode(raw, budget), owner)
    encoded.append((owner, raw))
    return tuple(encoded)


def _read_owner_receipt(connection, operation_id, source_fingerprint, owner, budget):
    """Validate one full committed candidate or the actual owning write snapshot."""
    if (
        owner not in _OWNERS
        or type(budget) is not _TraceReadBudget
        or not connection.in_transaction
    ):
        _invalid()
    if (
        type(operation_id) is not str
        or not 0 < len(operation_id) <= 256
        or not _hash(source_fingerprint)
    ):
        _invalid()
    budget.value((operation_id, source_fingerprint))
    prefix = owner + ".entry:"
    metadata = connection.execute(
        "SELECT count(*), coalesce(max(length(CAST(batch_key AS BLOB))),0), coalesce(max(length(CAST(source_fingerprint AS BLOB))),0), coalesce(max(length(CAST(mapping_json AS BLOB))),0), coalesce(sum(length(CAST(mapping_json AS BLOB))),0) FROM mobile_consolidation_batches WHERE operation_id=? AND substr(batch_key,1,?)=?",
        (operation_id, len(owner), owner),
    ).fetchone()
    count, key_size, source_size, maximum, total = metadata
    if not count:
        return None
    if key_size > 128 or source_size != 64 or maximum > _MAX_BYTES or maximum <= 0:
        _invalid()
    budget.container(count * 6)
    budget.reserve(total * 4 + count * (128 * 4 + 64 * 4))
    row = connection.execute(
        "SELECT source_fingerprint,mapping_json FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
        (operation_id, owner),
    ).fetchone()
    if row is None or row[0] != source_fingerprint:
        _invalid()
    master = _decode(row[1], budget)
    _validate_master(master, owner)
    if master["entry_count"] + 1 != count:
        _invalid()
    digest, children, previous, ordinal = hashlib.sha256(), [], None, 0
    cursor = connection.execute(
        "SELECT batch_key,source_fingerprint,mapping_json FROM mobile_consolidation_batches WHERE operation_id=? AND substr(batch_key,1,?)=? AND batch_key<>? ORDER BY length(batch_key),batch_key COLLATE BINARY",
        (operation_id, len(owner), owner, owner),
    )
    try:
        while True:
            budget.container(100)
            page = cursor.fetchmany(100)
            if not page:
                break
            for key, source, raw in page:
                if key != prefix + str(ordinal) or source != source_fingerprint:
                    _invalid()
                child = _decode(raw, budget)
                _validate_child(child, owner)
                data = _encode(child["data"], budget)
                budget.reserve(4 * len(data) + 64)
                ordering = (_KINDS.index(child["kind"]), data.encode("utf-8"))
                if previous is not None and ordering <= previous:
                    _invalid()
                previous = ordering
                _digest_entry(digest, key, raw, budget)
                children.append(child)
                ordinal += 1
    finally:
        cursor.close()
    if (
        ordinal != master["entry_count"]
        or digest.hexdigest() != master["entries_sha256"]
    ):
        _invalid()
    _validate_census(children, owner, budget)
    if owner == "runs":
        from .console_command_history import _validate_complete_history

        _validate_complete_history(
            connection,
            operation_id,
            source_fingerprint,
            master["command_history_count"],
            budget,
        )
    return master, tuple(children)


def _imported_definition_ids(connection, budget):
    """Complete every relevant candidate before returning any roster filter."""
    if not connection.in_transaction:
        _invalid()
    metadata = connection.execute(
        "SELECT count(*),coalesce(max(length(CAST(operation_id AS BLOB))),0),coalesce(max(length(CAST(source_fingerprint AS BLOB))),0) FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='runs'"
    ).fetchone()
    if metadata[1] > 256 or metadata[2] > 64:
        _invalid()
    budget.container(metadata[0] * 3)
    budget.reserve(metadata[0] * (256 * 4 + 64 * 4))
    cursor = connection.execute(
        "SELECT DISTINCT operation_id,source_fingerprint FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='runs' ORDER BY operation_id,source_fingerprint"
    )
    mapped = set()
    try:
        while True:
            page = cursor.fetchmany(100)
            if not page:
                break
            for operation, source in page:
                result = _read_owner_receipt(
                    connection, operation, source, "runs", budget
                )
                if result is None:
                    _invalid()
                for child in result[1]:
                    if (
                        child["kind"] == "identity"
                        and child["data"]["domain"] == "agent_definition"
                    ):
                        budget.container(1)
                        mapped.add(child["data"]["mapped"])
    finally:
        cursor.close()
    return mapped


def _row_digest(table, values, budget):
    """Hash exact manifest SQL scalars without serializing retained bodies to JSON."""
    spec = _SPECS.get(table)
    if (
        spec is None
        or len(values) != len(spec.columns)
        or any(column not in values for column in spec.columns)
    ):
        _invalid()
    digest = hashlib.sha256(b"tldw-mobile-row-v1\0")

    def framed(raw):
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)

    budget.reserve(4 * len(table) + 64)
    framed(table.encode("utf-8"))
    digest.update(struct.pack(">Q", len(spec.columns)))
    for column in spec.columns:
        budget.reserve(4 * len(column) + 128)
        framed(column.encode("utf-8"))
        value = values[column]
        if value is None:
            tag, raw = b"N", b""
        elif type(value) is int and _integer(value):
            tag, raw = b"I", str(value).encode("ascii")
        elif type(value) is float:
            tag, raw = b"R", struct.pack(">d", value)
        elif type(value) is str:
            budget.reserve(4 * len(value) + 64)
            tag, raw = b"T", value.encode("utf-8")
        elif type(value) is bytes:
            tag, raw = b"B", value
        else:
            _invalid()
        digest.update(tag)
        framed(raw)
    return digest.hexdigest()


def _imported_chat_records(connection, budget):
    """Complete the Chat census before current dispatch or fleet classification."""
    if not connection.in_transaction or type(budget) is not _TraceReadBudget:
        _invalid()
    metadata = connection.execute(
        "SELECT count(*),coalesce(max(length(CAST(operation_id AS BLOB))),0),coalesce(max(length(CAST(source_fingerprint AS BLOB))),0) FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='chat'"
    ).fetchone()
    if metadata[1] > 256 or metadata[2] > 64:
        _invalid()
    budget.container(metadata[0] * 4)
    budget.reserve(metadata[0] * (256 * 4 + 64 * 4))
    result = {"message": {}, "fleet_message": {}, "checkpoint": {}}
    cursor = connection.execute(
        "SELECT DISTINCT operation_id,source_fingerprint FROM mobile_consolidation_batches WHERE substr(batch_key,1,4)='chat' ORDER BY operation_id,source_fingerprint"
    )
    try:
        while True:
            page = cursor.fetchmany(100)
            if not page:
                break
            for operation, source in page:
                receipt = _read_owner_receipt(
                    connection, operation, source, "chat", budget
                )
                if receipt is None:
                    _invalid()
                children = receipt[1]
                budget.container(len(children) * 6)
                messages, checkpoints = {}, []
                for child in children:
                    data = child["data"]
                    if child["kind"] == "identity" and data["domain"] in {
                        "message",
                        "fleet_message",
                    }:
                        domain = data["domain"]
                        mapped = data["mapped"]
                        if mapped in result[domain]:
                            _invalid()
                        budget.container(4)
                        result[domain][mapped] = (operation, source)
                        if domain == "message":
                            messages[data["original"]] = mapped
                    elif (
                        child["kind"] == "row"
                        and data["table"] == "console_dispatch_checkpoints"
                    ):
                        checkpoints.append(data)
                # A checkpoint's PK is its actual assistant message identity.
                # Do not derive a discard action from an unrelated mapped row.
                for row in checkpoints:
                    original, mapped = row["original"][0], row["mapped"][0]
                    if (
                        messages.get(original) != mapped
                        or mapped in result["checkpoint"]
                    ):
                        _invalid()
                    budget.container(4)
                    result["checkpoint"][mapped] = (operation, source)
    finally:
        cursor.close()
    return result
