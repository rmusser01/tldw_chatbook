"""Closed normalized trace projection over actual captured native reader inputs."""

import hashlib
import json
from dataclasses import dataclass, field, replace

from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture
from tldw_chatbook.Chat.console_trace_models import (
    FrozenTracePolicy,
    TraceCallState,
    _validate_stable_opaque_id,
)
from tldw_chatbook.Chat.console_trace_projection import (
    ConsoleTraceProjection,
    NormalizedTraceCall,
)
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
from tldw_chatbook.Chat.console_trace_repository import HeaderComponentRef

from .mobile_content_graph import _ContentBlocker, _ContentRow, _UnprovedReference
from .mobile_content_memory import _OperationalProvenance
from .mobile_identity_plan import _DOMAINS
from .mobile_plan_manifest import _TABLES
from .mobile_trace_authority import _discharge_local_characters
from .mobile_trace_inputs import (
    _call_json,
    _check_trace_members,
    _check_trace_resources,
    _read_trace_inputs,
)

# These are the complete declared reference columns consumed by native readers.
# Literal content, ordering, masks, state and timestamps are never rewritten.
_REFERENCES = {
    "console_trace_artifacts": {"artifact_id": "trace_artifact"},
    "console_trace_segments": {
        "segment_id": "trace_segment",
        "parent_segment_id": "trace_segment",
        "inherited_surface_head_id": "trace_node",
    },
    "console_trace_owners": {
        "owner_id": "trace_owner",
        "conversation_id": "conversation",
        "root_segment_id": "trace_segment",
    },
    "console_trace_policies": {"policy_id": "trace_policy"},
    "console_trace_semantic_revisions": {
        "revision_id": "trace_revision",
        "source_conversation_id": "conversation",
        "source_message_id": "message",
        "predecessor_revision_id": "trace_revision",
        "live_message_id": "message",
    },
    "console_trace_revision_bindings": {
        "revision_id": "trace_revision",
        "policy_id": "trace_policy",
        "artifact_id": "trace_artifact",
    },
    "console_trace_surface_nodes": {
        "node_id": "trace_node",
        "segment_id": "trace_segment",
        "predecessor_node_id": "trace_node",
        "semantic_revision_id": "trace_revision",
        "artifact_id": "trace_artifact",
    },
    "console_trace_surface_replacements": {
        "replacement_id": "trace_replacement",
        "segment_id": "trace_segment",
        "predecessor_head_id": "trace_node",
        "start_node_id": "trace_node",
        "end_node_id": "trace_node",
        "replacement_node_id": "trace_node",
    },
    "console_trace_request_headers": {"header_id": "trace_header"},
    "console_trace_header_components": {
        "header_id": "trace_header",
        "artifact_id": "trace_artifact",
    },
    "console_trace_calls": {
        "call_id": "trace_call",
        "owner_id": "trace_owner",
        "segment_id": "trace_segment",
        "turn_id": "message",
        "run_id": "trace_group",
        "policy_id": "trace_policy",
        "surface_node_id": "trace_node",
        "request_header_id": "trace_header",
    },
    "console_trace_events": {
        "event_id": "trace_event",
        "segment_id": "trace_segment",
        "turn_id": "message",
        "call_id": "trace_call",
        "surface_node_id": "trace_node",
        "surface_replacement_id": "trace_replacement",
        "request_header_id": "trace_header",
        "semantic_revision_id": "trace_revision",
        "artifact_id": "trace_artifact",
    },
    "console_trace_response_links": {
        "response_link_id": "trace_response",
        "call_id": "trace_call",
        "semantic_revision_id": "trace_revision",
        "artifact_id": "trace_artifact",
    },
    "console_trace_redaction_spans": {
        "span_id": "trace_redaction",
        "policy_id": "trace_policy",
        "semantic_revision_id": "trace_revision",
        "artifact_id": "trace_artifact",
    },
    "console_trace_retention_roots": {"retention_id": "trace_retention"},
}
_OPERATIONAL = frozenset(
    {
        "console_trace_graph_epoch",
        "console_trace_migration_state",
        "console_trace_maintenance_state",
        "console_trace_compaction_state",
    }
)
_GC = frozenset(
    {
        "console_trace_gc_runs",
        "console_trace_gc_marks",
        "console_trace_gc_segment_scopes",
    }
)
_MESSAGE_FIELDS = (
    "sender",
    "role",
    "content",
    "image_data",
    "image_mime_type",
    "provider_continuation_json",
    "thinking_blocks_json",
    "assistant_generation_state",
)
_ATTACHMENT_FIELDS = ("position", "data", "mime_type", "display_name")


@dataclass(frozen=True)
class _TraceReconstruction:
    original_message_id: str
    mapped_message_id: str
    original_native: tuple[str, ...] = field(repr=False)
    mapped_native: tuple[str, ...] = field(repr=False)
    original_legacy: tuple[str, ...] = field(repr=False)
    mapped_legacy: tuple[str, ...] = field(repr=False)
    original_selected: tuple[str, ...] = field(repr=False)
    mapped_selected: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class _CitationDisposition:
    original: tuple
    source_manifest_digest: str
    canonical_manifest_digest: str
    disposition: str


@dataclass(frozen=True)
class _TraceInputIncidence:
    original_call_id: str
    mapped_call_id: str
    source_manifest_digest: str
    canonical_manifest_digest: str
    inputs: tuple[tuple[str, tuple], ...]


@dataclass(frozen=True)
class _TracePlan:
    content: object = field(repr=False)
    inputs: object = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    operational_provenance: tuple[_OperationalProvenance, ...]
    citation_dispositions: tuple[_CitationDisposition, ...]
    reconstructions: tuple[_TraceReconstruction, ...] = field(repr=False)
    field_ownership: tuple[tuple[str, tuple[str, ...]], ...]
    authority_discharges: tuple[_ContentRow, ...] = field(repr=False)
    input_incidence: tuple[_TraceInputIncidence, ...] = field(repr=False)
    pending_handlers: tuple[str, ...] = ("S", "A-runs", "integrated", "owner-write")

    @property
    def identities(self):
        return self.content.identities

    def require_complete(self):
        raise ValueError("mobile_trace_handlers_pending")


class _TraceBuilder:
    def __init__(self, content, inputs, budget, reads):
        self.reads = reads
        self.unproved_conversations = set()
        self.budget = budget
        budget.value(inputs.source.rows)
        budget.value(inputs.canonical.rows)
        self.content, self.inputs, self.identities = content, inputs, content.identities
        self.specs = {
            s.table: s for s in _TABLES if s.family == "chat" and s.handler == "T"
        }
        self.rows, self.blockers, self.provenance, self.citations = [], [], [], []
        self.originals = tuple(r for r in inputs.source.rows if r.table in self.specs)
        self.source = {
            table: [dict(r.values) for r in self.originals if r.table == table]
            for table in self.specs
        }
        self.canonical = {
            table: [dict(r.values) for r in inputs.canonical.rows if r.table == table]
            for table in self.specs
        }
        self.live_originals = {(r.table, r.original): r for r in content.source_rows}
        self.live_rows = {(r.table, r.original): r for r in content.rows}
        self.dependencies = []

    def key(self, table, row):
        return tuple(row[c] for c in self.specs[table].identity_columns)

    def block(self, table, row, reason, domain=None, identity=None):
        self.blockers.append(
            _ContentBlocker(table, self.key(table, row), reason, domain, identity)
        )

    def ref(self, domain, value):
        if value is None:
            return None
        try:
            mapped = self.identities.mapped(domain, value)
        except ValueError:
            raise _UnprovedReference(domain, value) from None
        self.dependencies.append((domain, value))
        if domain == "conversation" and value in self.unproved_conversations:
            raise _UnprovedReference(domain, value)
        table = {"message": "messages", "conversation": "conversations"}.get(domain)
        if (
            table
            and (table, (value,)) in self.live_originals
            and (table, (value,)) not in self.live_rows
        ):
            raise _UnprovedReference(domain, value)
        return mapped

    def transform(self, table, row):
        """Preserve native scalar grammar and transform only declared references."""
        refs = _REFERENCES[table]
        values = dict(row)
        for column, domain in refs.items():
            value = row[column]
            if (
                value is not None
                and domain.startswith("trace_")
                and domain != "trace_group"
            ):
                _validate_stable_opaque_id(value, column)
            values[column] = self.ref(domain, value)
        if table == "console_trace_artifacts":
            payload = row["sanitized_bytes"]
            if (
                type(payload) is not bytes
                or type(row["byte_length"]) is not int
                or row["byte_length"] != len(payload)
                or row["identity_digest"] != hashlib.sha256(payload).hexdigest()
            ):
                raise ValueError("native_artifact_bytes_invalid")
        elif table == "console_trace_policies":
            FrozenTracePolicy(
                row["policy_id"],
                row["credential_filter_version"],
                bool(row["pii_redaction_enabled"]),
                row["pii_ruleset_revision_id"],
            )
        elif table == "console_trace_header_components":
            HeaderComponentRef(
                row["component_kind"], row["ordinal"], row["artifact_id"]
            )
        elif table == "console_trace_request_headers":
            values["adapter_defaults_json"] = self.header_metadata(
                row["adapter_defaults_json"]
            )
        elif table == "console_trace_calls":
            TraceCallState(row["state"])
            if any(
                r["idempotency_key"] == row["idempotency_key"]
                for r in self.canonical[table]
            ):
                raise ValueError("native_call_idempotency_conflict")
        elif table == "console_trace_retention_roots":
            domain = {
                "owner": "trace_owner",
                "call": "trace_call",
                "revision": "trace_revision",
                "artifact": "trace_artifact",
            }.get(row["entity_kind"])
            if domain is None:
                raise ValueError("native_retention_kind_invalid")
            values["entity_id"] = self.ref(domain, row["entity_id"])
        return values

    def header_metadata(self, raw):
        # This is the sole structural JSON reference grammar. Artifact bytes and
        # matching strings in other metadata are historical content, never IDs.
        self.budget.json(raw)
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("native_header_metadata_invalid")
        composition = value.get("system_composition")
        if composition is None:
            return raw
        if type(composition) is not list or len(composition) > 256:
            raise ValueError("native_header_composition_invalid")
        mapped = []
        for token in composition:
            if type(token) is not dict:
                raise ValueError("native_header_composition_invalid")
            kind = token.get("kind")
            if kind == "revision":
                if set(token) != {"kind", "revision_id"}:
                    raise ValueError("native_header_composition_invalid")
                token = token | {
                    "revision_id": self.ref("trace_revision", token["revision_id"])
                }
            elif kind not in {
                "artifact",
                "omission",
                "transform_start",
                "transform_end",
            }:
                raise ValueError("native_header_composition_invalid")
            mapped.append(token)
        self.budget.value(value, text_copies=2)
        return json.dumps(
            value | {"system_composition": mapped},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def disposition(self, table, row):
        from .mobile_consolidation import _sqlite_row_digest

        if table in _GC:
            self.block(table, row, "native_gc_operation_unresolved")
        elif table in _OPERATIONAL:
            if (
                table.endswith("maintenance_state")
                and row["state"] != "idle"
                or table.endswith("compaction_state")
                and row["status"] == "running"
                or table.endswith("migration_state")
                and (
                    row["status"] in {"running", "failed"}
                    or row["last_exchange_id"] is not None
                )
            ):
                self.block(table, row, "native_trace_operation_unresolved")
            self.provenance.append(
                _OperationalProvenance(
                    table,
                    self.key(table, row),
                    _sqlite_row_digest(tuple(row.values())),
                    self.inputs.source.manifest_digest,
                    "retain_canonical_operational_authority",
                )
            )
        elif (
            table == "rag_identity_context"
            and self.inputs.source.citation.no_fingerprint_history
            and not any(
                self.source[t]
                for t in self.specs
                if t.startswith("rag_") and t != table
            )
        ):
            # No destination context row is emitted. SQL inertness does not assert
            # credential UNUSED, key readability or permission to adopt identity.
            self.citations.append(
                _CitationDisposition(
                    self.key(table, row),
                    self.inputs.source.manifest_digest,
                    self.inputs.canonical.manifest_digest,
                    "sql_no_history_context_not_adopted",
                )
            )
        else:
            self.block(
                table,
                row,
                "native_citation_namespace_readability_unproved",
                "citation_namespace",
                row.get("profile_id"),
            )

    def build(self):
        for table in self.specs:
            for row in self.source[table]:
                _check_trace_resources(self.reads)
                self.dependencies = []
                if table not in _REFERENCES:
                    self.disposition(table, row)
                    continue
                try:
                    values = self.transform(table, row)
                except _UnprovedReference as exc:
                    self.block(
                        table,
                        row,
                        "native_trace_dependency_unproved",
                        exc.domain,
                        exc.identity,
                    )
                    continue
                except (ValueError, TypeError, KeyError) as exc:
                    reason = (
                        str(exc)
                        if str(exc)
                        in {
                            "native_artifact_bytes_invalid",
                            "native_call_idempotency_conflict",
                            "native_header_metadata_invalid",
                            "native_header_composition_invalid",
                            "native_retention_kind_invalid",
                        }
                        else "native_trace_shape_invalid"
                    )
                    self.block(table, row, reason)
                    continue
                if set(values) != set(self.specs[table].columns):
                    raise ValueError("mobile_trace_field_ownership_mismatch")
                self.rows.append(
                    _ContentRow(
                        table,
                        self.key(table, row),
                        tuple(values.items()),
                        dependencies=tuple(self.dependencies),
                    )
                )
        self.close_dependencies()

    def close_dependencies(self):
        domains = {
            d.name: (d.table, d.column) for d in _DOMAINS if d.name != "trace_group"
        }
        while True:
            accepted = {(r.table, r.original) for r in self.rows}
            removed = []
            for row in self.rows:
                for domain, identity in row.dependencies:
                    target = domains.get(domain)
                    if (
                        target
                        and target[0] in _REFERENCES
                        and (target[0], (identity,)) not in accepted
                    ):
                        self.blockers.append(
                            _ContentBlocker(
                                row.table,
                                row.original,
                                "native_trace_dependency_unproved",
                                domain,
                                identity,
                            )
                        )
                        removed.append(row)
                        break
            if not removed:
                break
            self.rows = [r for r in self.rows if r not in removed]


class _TraceInputClosure:
    """The fixed reader's owned inputs, in addition to forward foreign keys."""

    def __init__(self, builder):
        self.builder = builder
        self.budget = builder.budget
        self.budget.container(len(builder.originals) * 4)
        self.original = {
            (row.table, row.original): dict(row.values) for row in builder.originals
        }
        self.accepted = {(row.table, row.original): row for row in builder.rows}
        self.domains = {
            domain.name: domain.table
            for domain in _DOMAINS
            if domain.name != "trace_group" and domain.table in _REFERENCES
        }
        self.messages = {
            row.original[0]: dict(row.values)
            for row in builder.inputs.source.rows
            if row.table == "messages"
        }

    def message(self, identity):
        original = self.messages.get(identity)
        if original is None:
            return  # A historical locator must not invent a live row.
        key = ("messages", (identity,))
        accepted = self.builder.live_rows.get(key)
        if accepted is None:
            raise _UnprovedReference("message", identity)
        mapped = dict(accepted.values)
        if any(mapped[name] != original[name] for name in _MESSAGE_FIELDS):
            raise _UnprovedReference("message", identity)
        for name, domain in (
            ("id", "message"),
            ("conversation_id", "conversation"),
            ("parent_message_id", "message"),
        ):
            value = original[name]
            if mapped[name] != (
                None if value is None else self.builder.identities.mapped(domain, value)
            ):
                raise _UnprovedReference("message", identity)
        self.remember(key)
        for row in self.builder.inputs.source.rows:
            if row.table != "message_attachments":
                continue
            values = dict(row.values)
            if values["message_id"] != identity:
                continue
            target = self.builder.live_rows.get((row.table, row.original))
            if target is None:
                raise _UnprovedReference("message_attachment", row.original)
            result = dict(target.values)
            if result["message_id"] != mapped["id"] or any(
                values[name] != result[name] for name in _ATTACHMENT_FIELDS
            ):
                raise _UnprovedReference("message_attachment", row.original)
            self.remember((row.table, row.original))

    def call(self, call):
        self.budget.container(2)
        self.seen = set()
        self.pending = []
        self.policy = call["policy_id"]
        self.add("console_trace_calls", call)
        self.surface(call["segment_id"], call["surface_node_id"], set())
        while self.pending:
            _check_trace_resources(self.builder.reads)
            table, key = self.pending.pop()
            original = self.original[(table, key)]
            accepted = self.accepted.get((table, key))
            if accepted is None:
                raise _UnprovedReference(table, key)
            # The transform records every fixed structural header reference too.
            for domain, identity in accepted.dependencies:
                if domain == "message":
                    self.message(identity)
                target = self.domains.get(domain)
                if target is not None:
                    self.add(target, self.original.get((target, (identity,))))
            if table == "console_trace_calls":
                self.owned("console_trace_events", "call_id", original["call_id"])
                self.owned(
                    "console_trace_response_links", "call_id", original["call_id"]
                )
            elif table == "console_trace_request_headers":
                self.owned(
                    "console_trace_header_components",
                    "header_id",
                    original["header_id"],
                )
            elif table == "console_trace_semantic_revisions":
                if original["live_message_id"] is not None:
                    self.message(original["live_message_id"])
                self.owned(
                    "console_trace_revision_bindings",
                    "revision_id",
                    original["revision_id"],
                    policy=True,
                )
                self.masks("semantic_revision_id", original["revision_id"])
            elif table == "console_trace_artifacts":
                self.masks("artifact_id", original["artifact_id"])

    def remember(self, key):
        if key in self.seen:
            return False
        self.budget.value(key)
        self.seen.add(key)
        return True

    def add(self, table, row):
        if row is None:
            return  # Original absence stays absent; native uncertainty is retained.
        key = (table, self.builder.key(table, row))
        if self.remember(key):
            self.budget.container(4)
            self.pending.append(key)

    def owned(self, table, column, identity, *, policy=False):
        for row in self.builder.source[table]:
            if row[column] == identity and (
                not policy or row["policy_id"] == self.policy
            ):
                self.add(table, row)

    def masks(self, column, identity):
        table = "console_trace_redaction_spans"
        rows = []
        for row in self.builder.source[table]:
            if row[column] == identity and row["policy_id"] == self.policy:
                self.add(table, row)
                self.budget.container()
                rows.append(row)
        # Frozen masks use the native SQL tie breaker; a remap cannot reorder it.
        self.order(
            table, rows, ("field_path", "start_codepoint", "end_codepoint", "span_id")
        )

    def order(self, table, rows, columns):
        self.budget.container(len(rows) * 4)
        original = sorted(rows, key=lambda row: tuple(row[name] for name in columns))
        mapped = []
        for row in rows:
            key = self.builder.key(table, row)
            accepted = self.accepted.get((table, key))
            if accepted is None:
                raise _UnprovedReference(table, key)
            mapped.append((key, dict(accepted.values)))
        mapped.sort(key=lambda item: tuple(item[1][name] for name in columns))
        if [self.builder.key(table, row) for row in original] != [
            key for key, _ in mapped
        ]:
            raise _UnprovedReference(table, self.builder.key(table, original[0]))

    def surface(self, segment_id, head_id, visited):
        # Match native prefix selection, without joining unrelated later calls
        # merely because they share a policy or a segment.
        pair = (segment_id, head_id)
        if pair in visited:
            raise _UnprovedReference("trace_segment", segment_id)
        self.budget.container()
        visited.add(pair)
        segment = self.original.get(("console_trace_segments", (segment_id,)))
        if segment is None or head_id is None:
            return
        head = self.original.get(("console_trace_surface_nodes", (head_id,)))
        if head is None:
            return
        self.add("console_trace_segments", segment)
        parent = segment["parent_segment_id"]
        inherited = segment["inherited_surface_head_id"]
        if parent is not None and inherited is not None:
            self.surface(parent, inherited, visited)
        nodes = set()
        for row in self.builder.source["console_trace_surface_nodes"]:
            if row["segment_id"] == segment_id and row["sequence"] <= head["sequence"]:
                self.add("console_trace_surface_nodes", row)
                self.budget.container()
                nodes.add(row["node_id"])
        replacements = []
        for row in self.builder.source["console_trace_surface_replacements"]:
            if row["segment_id"] == segment_id and row["replacement_node_id"] in nodes:
                self.add("console_trace_surface_replacements", row)
                self.budget.container()
                replacements.append(row)
        # Replacement node sequence is unique in a segment; equal-range ties
        # still retain their native stable-ID order.
        for node_id in nodes:
            self.budget.container(len(replacements))
            same_node = [
                row for row in replacements if row["replacement_node_id"] == node_id
            ]
            self.order(
                "console_trace_surface_replacements",
                same_node,
                ("start_sequence", "end_sequence", "replacement_id"),
            )


def _close_native_inputs(builder):
    closure = _TraceInputClosure(builder)
    refused = set()
    builder.budget.container()
    incidence = []
    for call in builder.source["console_trace_calls"]:
        key = ("console_trace_calls", (call["call_id"],))
        if key not in closure.accepted:
            continue
        try:
            closure.call(call)
            identities = builder.identities
            mapped = identities.mapped("trace_call", call["call_id"])
            builder.budget.container(5)
            for value in (
                call["call_id"],
                mapped,
                identities.source_manifest_digest,
                identities.canonical_manifest_digest,
            ):
                builder.budget.value(value)
            builder.budget.container(len(closure.seen) * 2)
            incidence.append(
                _TraceInputIncidence(
                    call["call_id"],
                    mapped,
                    identities.source_manifest_digest,
                    identities.canonical_manifest_digest,
                    tuple(sorted(closure.seen)),
                )
            )
        except _UnprovedReference as exc:
            builder.block(
                "console_trace_calls",
                call,
                "native_trace_input_closure_unproved",
                exc.domain,
                exc.identity,
            )
            refused.add(key)
    builder.rows = [
        row for row in builder.rows if (row.table, row.original) not in refused
    ]
    builder.close_dependencies()
    builder.budget.container(len(builder.rows))
    accepted = {(row.table, row.original) for row in builder.rows}
    builder.budget.container(len(incidence))
    return tuple(
        proof
        for proof in incidence
        if ("console_trace_calls", (proof.original_call_id,)) in accepted
    )


def _native_call(payload, budget):
    budget.json(payload)
    value = json.loads(payload)
    capture = value.pop("capture")
    capture["capture_detail"] = CaptureDetail(capture["capture_detail"])
    capture["omitted_keys"] = tuple(capture["omitted_keys"])
    capture["trace_uncertainty"] = tuple(capture["trace_uncertainty"])
    value.pop("source")
    value["uncertainty_codes"] = tuple(value["uncertainty_codes"])
    return NormalizedTraceCall(capture=ExchangeCapture(**capture), **value)


def _reconstructions(builder):
    """Qualify the fixed map against exact native per-input output associations."""
    inputs, identities = builder.inputs, builder.identities
    source_messages = {
        r.original[0]: dict(r.values)
        for r in inputs.source.rows
        if r.table == "messages"
    }
    mapped_messages = {
        r.original[0]: dict(r.values)
        for r in builder.content.rows
        if r.table == "messages"
    }
    accepted = {(r.table, r.original) for r in builder.rows}
    calls = {r["call_id"]: r for r in builder.source["console_trace_calls"]}
    result = []
    for message in inputs.source.messages:
        _check_trace_resources(builder.reads)
        original = source_messages[message.message_id]
        mapped = mapped_messages.get(message.message_id)
        if mapped is None or any(mapped[k] != original[k] for k in _MESSAGE_FIELDS):
            builder.blockers.append(
                _ContentBlocker(
                    "messages",
                    (message.message_id,),
                    "native_trace_message_projection_changed",
                )
            )
            continue
        if (
            mapped["id"] != identities.mapped("message", original["id"])
            or mapped["conversation_id"]
            != identities.mapped("conversation", original["conversation_id"])
            or mapped["parent_message_id"]
            != (
                None
                if original["parent_message_id"] is None
                else identities.mapped("message", original["parent_message_id"])
            )
        ):
            raise ValueError("mobile_trace_message_map_invalid")
        attachments = [
            r
            for r in inputs.source.rows
            if r.table == "message_attachments"
            and dict(r.values)["message_id"] == message.message_id
        ]
        mapped_attachments = {
            r.original: dict(r.values)
            for r in builder.content.rows
            if r.table == "message_attachments"
        }
        if any(
            r.original not in mapped_attachments
            or any(
                dict(r.values)[k] != mapped_attachments[r.original][k]
                for k in _ATTACHMENT_FIELDS
            )
            for r in attachments
        ):
            builder.blockers.append(
                _ContentBlocker(
                    "messages",
                    (message.message_id,),
                    "native_trace_attachment_projection_changed",
                )
            )
            continue
        originals = [
            tuple(_native_call(payload, builder.budget) for payload in message.native),
            tuple(_native_call(payload, builder.budget) for payload in message.legacy),
        ]
        if any(
            ("console_trace_calls", (call.call_id,)) not in accepted
            for group in originals
            for call in group
        ):
            continue
        mapped_groups = []
        for group in originals:
            projected = []
            for call in group:
                projected.append(
                    replace(
                        call,
                        call_id=identities.mapped("trace_call", call.call_id),
                        capture=replace(
                            call.capture,
                            run_tag=identities.mapped(
                                "trace_group", calls[call.call_id]["run_id"]
                            ),
                        )
                        if call.provenance == "native"
                        else call.capture,
                    )
                )
            mapped_groups.append(tuple(projected))
        # Legacy raw order is a native SQL contract distinct from viewer order.
        pairs = zip(originals[1], mapped_groups[1], strict=True)
        mapped_groups[1] = tuple(
            mapped
            for original, mapped in sorted(
                pairs,
                key=lambda pair: (
                    identities.mapped("trace_group", calls[pair[0].call_id]["run_id"]),
                    calls[pair[0].call_id]["call_sequence"],
                    pair[1].call_id,
                ),
            )
        )
        builder.budget.value(mapped_groups)
        combined = mapped_groups[0] + mapped_groups[1]
        selected = ConsoleTraceProjection(
            legacy_reader=lambda _: (),
            normalized_reader=lambda _, calls=combined: calls,
            normalized_reads_enabled=True,
        ).read_calls(identities.mapped("message", message.message_id))
        expected = tuple(
            identities.mapped(
                "trace_call", _native_call(payload, builder.budget).call_id
            )
            for payload in message.selected
        )
        if tuple(call.call_id for call in selected) != expected:
            builder.blockers.append(
                _ContentBlocker(
                    "messages",
                    (message.message_id,),
                    "native_trace_projection_selection_changed",
                )
            )
            continue
        result.append(
            _TraceReconstruction(
                message.message_id,
                mapped["id"],
                message.native,
                tuple(_call_json(call, builder.budget) for call in mapped_groups[0]),
                message.legacy,
                tuple(_call_json(call, builder.budget) for call in mapped_groups[1]),
                message.selected,
                tuple(_call_json(call, builder.budget) for call in selected),
            )
        )
    return tuple(result)


def _plan_traces(reads, *, _read_budget=None):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_trace_native_snapshot_required")
    reads.check()
    content = reads.content_plan()
    budget = _TraceReadBudget() if _read_budget is None else _read_budget
    inputs = _read_trace_inputs(reads, _read_budget=budget)
    if (
        content.identities.source_manifest_digest != inputs.source.manifest_digest
        or content.identities.canonical_manifest_digest
        != inputs.canonical.manifest_digest
    ):
        raise ValueError("mobile_trace_input_binding_changed")
    authority_discharges, authority_blockers = _discharge_local_characters(
        reads, content, inputs, budget
    )
    builder = _TraceBuilder(content, inputs, budget, reads)
    builder.unproved_conversations.update(b.original[0] for b in authority_blockers)
    builder.blockers.extend(authority_blockers)
    builder.build()
    incidence = _close_native_inputs(builder)
    reconstructions = _reconstructions(builder)
    _check_trace_members(reads)
    return _TracePlan(
        content,
        inputs,
        tuple(builder.rows),
        builder.originals,
        tuple(builder.blockers),
        tuple(builder.provenance),
        tuple(builder.citations),
        reconstructions,
        tuple((t, s.columns) for t, s in builder.specs.items()),
        authority_discharges,
        incidence,
    )
