"""Private typed identity reservations; no allocation or imported owner writes."""

import hashlib
import json
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class _Domain:
    name: str
    family: str
    table: str
    column: str
    scalar: type = str
    repeated: bool = False


# Each declaration is a native ownership identity, not a spelling heuristic.
_DOMAINS = (
    _Domain("workspace", "workspaces", "workspace_records", "workspace_id"),
    _Domain("binding", "workspaces", "workspace_runtime_bindings", "binding_id"),
    _Domain("membership", "workspaces", "workspace_memberships", "membership_id"),
    _Domain("workspace_audit", "workspaces", "workspace_handoff_audit", "audit_id"),
    _Domain("conversation", "chat", "conversations", "id"),
    _Domain("message", "chat", "messages", "id"),
    _Domain("character", "chat", "character_cards", "id", int),
    _Domain("note", "chat", "notes", "id"),
    _Domain("keyword", "chat", "keywords", "id", int),
    _Domain("keyword_collection", "chat", "keyword_collections", "id", int),
    _Domain("dictionary", "chat", "chat_dictionaries", "id", int),
    _Domain("world_book", "chat", "world_books", "id", int),
    _Domain("world_book_entry", "chat", "world_book_entries", "id", int),
    _Domain("note_folder", "chat", "note_folders", "id"),
    _Domain("note_folder_membership", "chat", "note_folder_memberships", "id"),
    _Domain("canvas", "chat", "canvas_documents", "id"),
    _Domain("canvas_revision", "chat", "canvas_revisions", "id"),
    _Domain("buddy", "chat", "buddy_profiles", "id"),
    _Domain("character_expression", "chat", "character_expression_images", "id", int),
    _Domain("buddy_visual_binding", "chat", "buddy_visual_bindings", "id", int),
    _Domain("persona_visual_binding", "chat", "persona_visual_bindings", "id", int),
    _Domain("persona_visual_pack", "chat", "persona_visual_packs", "id", int),
    _Domain(
        "persona_visual_version", "chat", "persona_visual_pack_versions", "id", int
    ),
    _Domain("persona_visual_asset", "chat", "persona_visual_assets", "id", int),
    _Domain("visual_binding", "chat", "visual_identity_bindings", "id", int),
    _Domain("visual_pack", "chat", "visual_identity_packs", "id", int),
    _Domain("visual_version", "chat", "visual_identity_pack_versions", "id", int),
    _Domain("visual_asset", "chat", "visual_identity_assets", "id", int),
    _Domain(
        "response_rule",
        "chat",
        "console_response_rule_revisions",
        "rule_id",
        repeated=True,
    ),
    _Domain(
        "response_assessment",
        "chat",
        "console_response_rule_assessments",
        "assessment_id",
    ),
    _Domain("response_draft", "chat", "console_response_rule_drafts", "draft_id"),
    _Domain("transcript_annotation", "chat", "transcript_annotations", "annotation_id"),
    _Domain("fleet_message", "chat", "fleet_progress_messages", "message_id"),
    _Domain("memory", "chat", "console_conversation_memories", "id"),
    _Domain(
        "memory_selection",
        "chat",
        "console_conversation_memory_selections",
        "selection_id",
    ),
    _Domain("trace_owner", "chat", "console_trace_owners", "owner_id"),
    _Domain("trace_segment", "chat", "console_trace_segments", "segment_id"),
    _Domain("trace_policy", "chat", "console_trace_policies", "policy_id"),
    _Domain("trace_artifact", "chat", "console_trace_artifacts", "artifact_id"),
    _Domain("trace_call", "chat", "console_trace_calls", "call_id"),
    _Domain("trace_group", "chat", "console_trace_calls", "run_id", repeated=True),
    _Domain("trace_event", "chat", "console_trace_events", "event_id"),
    _Domain("trace_node", "chat", "console_trace_surface_nodes", "node_id"),
    _Domain(
        "trace_replacement",
        "chat",
        "console_trace_surface_replacements",
        "replacement_id",
    ),
    _Domain("trace_header", "chat", "console_trace_request_headers", "header_id"),
    _Domain(
        "trace_revision", "chat", "console_trace_semantic_revisions", "revision_id"
    ),
    _Domain(
        "trace_response", "chat", "console_trace_response_links", "response_link_id"
    ),
    _Domain("trace_redaction", "chat", "console_trace_redaction_spans", "span_id"),
    _Domain("trace_retention", "chat", "console_trace_retention_roots", "retention_id"),
    _Domain("auxiliary_attempt", "chat", "console_auxiliary_attempts", "operation_id"),
    _Domain("run", "runs", "agent_runs", "id"),
    _Domain("agent_definition", "runs", "agent_definitions", "id"),
    _Domain("work_chain", "runs", "automatic_work_chains", "id"),
    _Domain("work_reservation", "runs", "automatic_work_reservations", "id"),
    _Domain("wake_attempt", "runs", "automatic_wake_attempts", "id"),
    _Domain("chat_start_attempt", "runs", "automatic_chat_start_attempts", "id"),
    _Domain("change_snapshot", "runs", "change_snapshots", "id", int),
    _Domain("change_note", "runs", "change_notes", "id", int),
    _Domain("activity", "runs", "console_activity_receipts", "activity_id"),
    _Domain("command", "runs", "console_command_receipts", "command_id"),
    _Domain("goal", "runs", "goal_runs", "id"),
    _Domain("goal_iteration", "runs", "goal_iterations", "id"),
    _Domain("goal_report", "runs", "goal_reports", "id"),
    _Domain("goal_evidence", "runs", "goal_evidence", "id"),
)

# Segment/owner-local order is retained; only the declared owning identity maps.
_COMPOSITES = (
    (
        "response_rule_revision",
        "chat",
        "console_response_rule_revisions",
        (("rule_id", "response_rule"), ("revision", int)),
    ),
    (
        "response_rule_fixture",
        "chat",
        "console_response_rule_fixtures",
        (("rule_id", "response_rule"), ("revision", int), ("case_id", str)),
    ),
    (
        "response_rule_validation",
        "chat",
        "console_response_rule_validations",
        (("rule_id", "response_rule"), ("revision", int)),
    ),
    (
        "canvas_position",
        "chat",
        "canvas_revisions",
        (("canvas_id", "canvas"), ("sequence", int)),
    ),
    ("run_step", "runs", "agent_run_steps", (("run_id", "run"), ("seq", int))),
    (
        "trace_header_component",
        "chat",
        "console_trace_header_components",
        (("header_id", "trace_header"), ("component_kind", str), ("ordinal", int)),
    ),
    (
        "trace_revision_binding",
        "chat",
        "console_trace_revision_bindings",
        (("revision_id", "trace_revision"), ("policy_id", "trace_policy")),
    ),
    (
        "message_attachment",
        "chat",
        "message_attachments",
        (("message_id", "message"), ("position", int)),
    ),
    (
        "message_generation",
        "chat",
        "message_generation_metadata",
        (("message_id", "message"), ("position", int)),
    ),
    (
        "message_trajectory",
        "chat",
        "message_trajectory_metadata",
        (("message_id", "message"), ("seq", int), ("event_kind", str)),
    ),
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


def _reserve(domain, source, canonical, fingerprint):
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
            if domain.name == "world_book_entry":
                existing = {row[domain.column] for row in rows}
                rows += tuple(
                    {domain.column: value}
                    for value in sorted(
                        _retired_world_entries(reads, purpose) - existing
                    )
                )
            partitions.append(rows)
        reservations.extend(_reserve(domain, *partitions, fingerprint))
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
