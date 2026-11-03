"""Private typed identity reservations; no allocation or imported owner writes."""

import hashlib
import json
from dataclasses import dataclass
from uuid import UUID

from tldw_chatbook.DB.mobile_import_authority import (
    _COMPOSITES,
    _DOMAINS,
)
from tldw_chatbook.DB.mobile_import_authority import (
    _Domain as _Domain,  # noqa: PLC0414 - canonical compatibility export
)


@dataclass(frozen=True)
class _Reservation:
    domain: str
    original: str | int | tuple
    mapped: str | int | tuple


# These keys carry native citation authority namespaces, not relocatable UUIDs.
# T must prove the native namespace projection before integrated allocation.
_CITATION_IDENTITIES = (
    (
        "rag_identity_context",
        (
            ("context_name", str),
            ("profile_id", str),
            ("local_authority_id", str),
            ("fingerprint_key_id", str),
        ),
    ),
    ("rag_citation_traces", (("profile_id", str), ("trace_id", str))),
    ("rag_answer_attempt_payloads", (("profile_id", str), ("payload_id", str))),
    ("rag_evidence_snapshots", (("profile_id", str), ("payload_id", str))),
    ("rag_evidence_runs", (("profile_id", str), ("trace_id", str), ("run_id", str))),
    (
        "rag_trace_evidence_refs",
        (
            ("profile_id", str),
            ("trace_id", str),
            ("prompt_set_id", str),
            ("evidence_ordinal", int),
        ),
    ),
    (
        "rag_message_trace_owners",
        (
            ("profile_id", str),
            ("message_id", str),
            ("message_revision", int),
            ("trace_id", str),
        ),
    ),
    (
        "rag_source_observations",
        (
            ("profile_id", str),
            ("trace_id", str),
            ("prompt_set_id", str),
            ("evidence_ordinal", int),
            ("snapshot_payload_id", str),
            ("resolver_kind", str),
            ("resolver_version", str),
        ),
    ),
    (
        "rag_artifact_owner_leases",
        (
            ("profile_id", str),
            ("artifact_store_id", str),
            ("artifact_id", str),
            ("artifact_revision", int),
            ("trace_id", str),
        ),
    ),
    ("rag_artifact_owner_operations", (("profile_id", str), ("operation_id", str))),
    ("rag_legacy_migration_journal", (("profile_id", str), ("conversation_id", str))),
    (
        "rag_payload_tombstones",
        (("profile_id", str), ("origin_namespace", str), ("origin_payload_id", str)),
    ),
)


@dataclass(frozen=True)
class _PendingIdentity:
    handler: str
    family: str
    table: str
    original: tuple[str | int, ...]


@dataclass(frozen=True)
class _IdentityPlan:
    source_fingerprint: str
    source_manifest_digest: str
    canonical_manifest_digest: str
    reservations: tuple[_Reservation, ...]
    pending_dependencies: tuple[_PendingIdentity, ...] = ()

    def require_resolved(self):
        if self.pending_dependencies:
            raise ValueError("mobile_identity_dependencies_pending")

    def mapped(self, domain: str, identity: str | int | tuple):
        for row in self.reservations:
            if (
                row.domain == domain
                and _same_type(identity, row.original)
                and identity == row.original
            ):
                return row.mapped
        raise ValueError("mobile_identity_reference_unresolved:" + domain)


def _same_type(left, right):
    return type(left) is type(right) and (
        type(left) is not tuple
        or len(left) == len(right)
        and all(_same_type(a, b) for a, b in zip(left, right, strict=True))
    )


def _ordered(reads, domain, table):
    source = _rows(reads, "legacy", "chat", table)
    canonical = _rows(reads, "canonical", "chat", table)
    values = [row["sequence"] for row in source]
    existing = [row["sequence"] for row in canonical]
    frontiers = [
        row["seq"]
        for row in _rows(reads, "canonical", "chat", "sqlite_sequence")
        if row["name"] == table
    ]
    if any(
        type(value) is not int or value < 0 for value in values + existing + frontiers
    ) or len(set(values)) != len(values):
        raise ValueError("mobile_identity_order_invalid:" + domain)
    frontier = max(existing + frontiers, default=0)
    if frontier + len(values) > 2**63 - 1:
        raise ValueError("mobile_identity_order_exhausted:" + domain)
    return tuple(
        _Reservation(domain, value, frontier + index)
        for index, value in enumerate(sorted(values), 1)
    )


def _rows(reads, purpose, family, table, *, _read_budget=None):
    from .mobile_captured_sqlite import _runs24_receipts_absent

    if _runs24_receipts_absent(reads, purpose, family, table):
        return ()
    if not any(p == purpose and f == family for p, f, _ in reads.versions):
        return ()
    if _read_budget is not None:
        _read_budget.container()
    result, offset = [], 0
    while True:
        columns, rows = reads.page(purpose, family, table, offset=offset, limit=1000)
        for row in rows:
            if _read_budget is not None:
                _read_budget.value(row)
                _read_budget.container(len(columns) * 2)
            result.append(dict(zip(columns, row, strict=True)))
        if len(rows) < 1000:
            if _read_budget is not None:
                _read_budget.container(len(result))
            return tuple(result)
        offset += len(rows)


def _reserve(domain, source, canonical, fingerprint, *, _read_budget=None):
    if _read_budget is not None:
        _read_budget.container((len(source) + len(canonical)) * 12)
        for row in (*source, *canonical):
            _read_budget.value(row[domain.column], text_copies=2)
    original = [row[domain.column] for row in source]
    existing = [row[domain.column] for row in canonical]
    if any(type(value) is not domain.scalar for value in original + existing):
        raise ValueError("mobile_identity_type_invalid:" + domain.name)
    if domain.name == "trace_group":
        from tldw_chatbook.Chat.console_trace_repository import _nonempty

        for identity in original + existing:
            _nonempty(identity, "run_id")
    if domain.repeated:
        # A rule owns several revision rows; its logical identity reserves once.
        original, existing = list(set(original)), list(set(existing))
    if domain.name in {"canvas", "canvas_revision"}:
        from tldw_chatbook.Canvas.repository import _validated_uuid

        for identity in original + existing:
            _validated_uuid(identity, domain.name)
    if len(set(original)) != len(original) or len(set(existing)) != len(existing):
        raise ValueError("mobile_identity_ambiguous:" + domain.name)
    occupied = set(original) | set(existing)
    next_integer = max(occupied, default=0) + 1 if domain.scalar is int else None
    for identity in sorted(original):
        target = identity
        if identity in existing and not (
            domain.name == "workspace" and identity == "workspace-default"
        ):
            if domain.scalar is int:
                if next_integer > 2**63 - 1:
                    raise ValueError("mobile_identity_integer_exhausted:" + domain.name)
                target, next_integer = next_integer, next_integer + 1
            else:
                attempt = 0
                while True:
                    if _read_budget is not None:
                        _read_budget.value(
                            (fingerprint, domain.name, identity, attempt), text_copies=3
                        )
                    seed = json.dumps(
                        (fingerprint, domain.name, identity, attempt),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    # Native trace records require the canonical UUIDv4 shape.
                    target = str(
                        UUID(bytes=hashlib.sha256(seed).digest()[:16], version=4)
                    )
                    if target not in occupied:
                        break
                    attempt += 1
            if domain.name.startswith("trace_") and domain.scalar is str:
                from tldw_chatbook.Chat.console_trace_models import _validate_opaque_id

                _validate_opaque_id(target, domain.name)
            if domain.name in {"canvas", "canvas_revision"}:
                _validated_uuid(target, domain.name)
            occupied.add(target)
        yield _Reservation(domain.name, identity, target)


def _historical_locators(reads, purpose):
    """Retained revision locators share live domains without creating live owners."""
    messages = _rows(reads, purpose, "chat", "messages")
    revisions = _rows(reads, purpose, "chat", "console_trace_semantic_revisions")
    owners = {}
    sequences = set()
    revision_by_id = {row["revision_id"]: row for row in revisions}
    for row in messages:
        owners[row["id"]] = row["conversation_id"]
    for row in revisions:
        message, conversation = row["source_message_id"], row["source_conversation_id"]
        if any(
            type(value) is not str or not value for value in (message, conversation)
        ):
            raise ValueError("mobile_identity_historical_locator_invalid")
        if message in owners and owners[message] != conversation:
            raise ValueError("mobile_identity_historical_owner_ambiguous")
        owners[message] = conversation
        if row["live_message_id"] is not None and row["live_message_id"] != message:
            raise ValueError("mobile_identity_historical_live_locator_invalid")
        sequence = row["revision_sequence"]
        key = (conversation, message, sequence)
        if type(sequence) is not int or sequence < 0 or key in sequences:
            raise ValueError("mobile_identity_historical_sequence_invalid")
        sequences.add(key)
        predecessor = row["predecessor_revision_id"]
        if predecessor is not None:
            prior = revision_by_id.get(predecessor)
            if (
                prior is None
                or (prior["source_message_id"], prior["source_conversation_id"])
                != (message, conversation)
                or prior["revision_sequence"] >= sequence
            ):
                raise ValueError("mobile_identity_historical_predecessor_invalid")
    return {
        "message": {row["source_message_id"] for row in revisions},
        "conversation": {row["source_conversation_id"] for row in revisions},
    }


def _retired_world_entry(row):
    """Validate the native hard-delete proof without recreating a live owner."""
    from .mobile_workspace_plan import _object

    try:
        payload = _object(row["payload"])
        if (
            row["entity"] != "world_book_entries"
            or row["operation"] != "delete"
            or type(row["version"]) is not int
            or row["version"] != 1
            or set(payload) != {"id", "world_book_id"}
            or any(
                type(v) is not int or not -(2**63) <= v < 2**63
                for v in payload.values()
            )
            or type(row["entity_id"]) is not str
            or row["entity_id"] != str(payload["id"])
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "mobile_retired_entry_proof_invalid:" + str(row["change_id"])
        ) from None
    return payload


def _retired_world_entries(reads, purpose):
    return {
        _retired_world_entry(row)["id"]
        for row in _rows(reads, purpose, "chat", "sync_log")
        if row["entity"] == "world_book_entries" and row["operation"] == "delete"
    }


def _plan_identities(reads):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_identity_native_snapshot_required")
    reads.check()
    captured = reads.captured
    fingerprint = captured.preview.source_fingerprint
    reservations = []
    historical = {
        purpose: _historical_locators(reads, purpose)
        for purpose in ("legacy", "canonical")
    }
    for domain in _DOMAINS:
        partitions = []
        for purpose in ("legacy", "canonical"):
            rows = _rows(reads, purpose, domain.family, domain.table)
            if domain.name in {"message", "conversation"}:
                existing = {row[domain.column] for row in rows}
                rows += tuple(
                    {domain.column: value}
                    for value in sorted(historical[purpose][domain.name] - existing)
                )
            if domain.name == "binding":
                budget = reads.read_budget
                witnesses = _rows(
                    reads,
                    purpose,
                    "workspaces",
                    "workspace_binding_authority",
                    _read_budget=budget,
                )
                budget.container((len(rows) + len(witnesses)) * 8)
                if any(
                    type(r["revision"]) is not int
                    or not 1 <= r["revision"] <= 2**63 - 1
                    for r in witnesses
                ):
                    raise ValueError("mobile_binding_witness_invalid")
                existing = {r[domain.column] for r in rows}
                witnessed = {r["binding_id"] for r in witnesses}
                if (
                    purpose,
                    "workspaces",
                    8,
                ) not in reads.versions and not existing <= witnessed:
                    raise ValueError("mobile_binding_witness_missing")
                rows += tuple({domain.column: value} for value in witnessed - existing)
            if domain.name == "world_book_entry":
                existing = {row[domain.column] for row in rows}
                rows += tuple(
                    {domain.column: value}
                    for value in sorted(
                        _retired_world_entries(reads, purpose) - existing
                    )
                )
            partitions.append(rows)
        reservations.extend(
            _reserve(
                domain,
                *partitions,
                fingerprint,
                _read_budget=reads.read_budget if domain.name == "binding" else None,
            )
        )
    for domain, table in (
        ("memory_sequence", "console_conversation_memory_selections"),
        ("fleet_sequence", "fleet_progress_messages"),
    ):
        reservations.extend(_ordered(reads, domain, table))
    plan = _IdentityPlan(
        fingerprint,
        hashlib.sha256(captured.source.manifest_bytes).hexdigest(),
        hashlib.sha256(captured.canonical.manifest_bytes).hexdigest(),
        tuple(reservations),
    )
    for name, family, table, fields in _COMPOSITES:
        for row in _rows(reads, "legacy", family, table):
            original, mapped = [], []
            for column, kind in fields:
                value = row[column]
                original.append(value)
                if isinstance(kind, str):
                    mapped.append(plan.mapped(kind, value))
                elif type(value) is kind:
                    mapped.append(value)
                else:
                    raise ValueError("mobile_identity_type_invalid:" + name)
            reservations.append(_Reservation(name, tuple(original), tuple(mapped)))
    pending = []
    for table, fields in _CITATION_IDENTITIES:
        for row in _rows(reads, "legacy", "chat", table):
            if any(type(row[column]) is not scalar for column, scalar in fields):
                raise ValueError("mobile_identity_type_invalid:" + table)
            pending.append(
                _PendingIdentity(
                    "T", "chat", table, tuple(row[column] for column, _ in fields)
                )
            )
    reads.check()
    return _IdentityPlan(
        plan.source_fingerprint,
        plan.source_manifest_digest,
        plan.canonical_manifest_digest,
        tuple(reservations),
        tuple(pending),
    )
