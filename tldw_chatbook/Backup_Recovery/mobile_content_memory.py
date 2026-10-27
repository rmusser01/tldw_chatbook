"""Fixed C3 history proofs and complete, noninsertable Console content plans."""

import json
from dataclasses import dataclass, field, replace
from itertools import pairwise

from .mobile_content_actors import _ActorPlan
from .mobile_content_graph import _ContentBlocker, _ContentRow, _GraphBuilder
from .mobile_content_settings import _USAGE_JSON, _SettingsSuccessor, _typed
from .mobile_content_workflows import _integer, _shape, _text
from .mobile_workspace_plan import _object

_SEARCH_TABLES = (
    "character_conversation_search_dirty",
    "character_conversation_search_documents",
    "character_conversation_search_generations",
    "character_conversation_search_revision",
    "character_conversation_search_state",
)
_TABLES = _SEARCH_TABLES + (
    "console_auxiliary_attempts",
    "console_conversation_context_policy",
    "console_conversation_memories",
    "console_conversation_memory_scopes",
    "console_conversation_memory_selections",
)


@dataclass(frozen=True)
class _OperationalProvenance:
    table: str
    original: tuple[str | int, ...]
    source_digest: str
    source_manifest_digest: str
    disposition: str


@dataclass(frozen=True)
class _ContentPlan:
    actors: _ActorPlan = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    operational_provenance: tuple[_OperationalProvenance, ...]
    successors: tuple[_SettingsSuccessor, ...]
    discharged: tuple[_SettingsSuccessor, ...]
    field_ownership: tuple[tuple[str, tuple[str, ...]], ...]
    pending_handlers: tuple[str, ...] = ("T", "S", "A-runs")

    @property
    def identities(self):
        return self.actors.identities

    @property
    def authority_slots(self):
        return self.actors.authority_slots

    @property
    def dictionary_derivations(self):
        return self.actors.dictionary_derivations

    def require_complete(self) -> None:
        raise ValueError("mobile_content_handlers_pending")


def _context_and_accounting(builder):
    from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
    from tldw_chatbook.Chat.console_context_repository import (
        _FAILURE_REASON_RE,
        AuxiliaryAttemptStart,
        AuxiliaryAttemptStatus,
        AuxiliaryPricingProvenance,
    )

    def policy(row):
        ConsoleContextPolicyOverrides.from_mapping(row)
        _integer(row["policy_revision"], 1)
        _text(row["updated_at"], limit=80)
        builder.emit(
            "console_conversation_context_policy",
            row,
            row
            | {"conversation_id": builder.ref("conversation", row["conversation_id"])},
        )

    builder.handle("console_conversation_context_policy", policy)

    def auxiliary(row):
        AuxiliaryAttemptStart(
            **{key: row[key] for key in AuxiliaryAttemptStart.__dataclass_fields__}
        )
        status = AuxiliaryAttemptStatus(row["status"])
        _text(row["finished_at"], optional=True, limit=80)
        if row["elapsed_ms"] is not None:
            _integer(row["elapsed_ms"])
        if row["failure_reason"] is not None and (
            status is AuxiliaryAttemptStatus.SUCCEEDED
            or _FAILURE_REASON_RE.fullmatch(row["failure_reason"]) is None
        ):
            raise ValueError("invalid_auxiliary_reason")
        if status is AuxiliaryAttemptStatus.STARTED:
            if any(
                row[key] is not None
                for key in (
                    "finished_at",
                    "elapsed_ms",
                    "provider_usage_json",
                    "pricing_provenance_json",
                    "failure_reason",
                )
            ):
                raise ValueError("invalid_started_attempt")
        elif row["finished_at"] is None:
            raise ValueError("missing_auxiliary_finish")
        if row["pricing_provenance_json"] is not None:
            pricing = _object(row["pricing_provenance_json"])
            _shape(pricing, ("catalog_revision", "source", "estimated"))
            AuxiliaryPricingProvenance(**pricing)
        if row["provider_usage_json"] is not None:
            usage = _typed(_USAGE_JSON, row["provider_usage_json"])
            if any(
                getattr(usage, field) < 0
                for field in (
                    "uncached_input",
                    "cache_read",
                    "cache_write",
                    "output",
                    "audio_input",
                    "audio_output",
                    "transcription_seconds",
                )
            ):
                raise ValueError("negative_usage")
        builder.emit(
            "console_auxiliary_attempts",
            row,
            row
            | {
                "operation_id": builder.ref("auxiliary_attempt", row["operation_id"]),
                "conversation_id": builder.ref("conversation", row["conversation_id"]),
            },
        )

    builder.handle("console_auxiliary_attempts", auxiliary)


def _search(builder):
    """Retain exact source associations; native destination indexing owns effects."""
    from .mobile_consolidation import _sqlite_row_digest

    provenance = []
    for table in _SEARCH_TABLES[-2:]:
        rows = builder.canonical[table]
        if len(rows) != 1 or rows[0]["singleton_id"] != 1:
            builder.blockers.append(
                _ContentBlocker(table, (), "canonical_native_singleton_invalid")
            )
    for row in builder.canonical["character_conversation_search_generations"]:
        if row["status"] == "building":
            builder.blockers.append(
                _ContentBlocker(
                    "character_conversation_search_generations",
                    (row["generation_id"],),
                    "canonical_search_build_unresolved",
                )
            )
    for table in _SEARCH_TABLES:
        source = builder.source[table]
        if table.endswith(("_state", "_revision")) and (
            len(source) != 1 or source[0]["singleton_id"] != 1
        ):
            builder.blockers.append(
                _ContentBlocker(table, (), "native_singleton_invalid")
            )
        for row in source:
            original = builder.original(table, row)
            try:
                if table.endswith("_revision"):
                    _integer(row["data_revision"])
                elif table.endswith("_state"):
                    _integer(row["activated"], 0, 1)
                    _text(row["data_authority_id"], optional=True, limit=200)
                    if row["active_policy_version"] is not None:
                        _integer(row["active_policy_version"], 1)
                    if row["activated"] and (
                        row["data_authority_id"] is None
                        or row["active_policy_version"] is None
                    ):
                        raise ValueError("search_activation_invalid")
                else:
                    _integer(row["source_revision"])
                    _text(row["data_authority_id"], limit=200)
                if table.endswith("_generations"):
                    _text(row["generation_id"], limit=200)
                    _integer(row["policy_version"], 1)
                    _integer(row["processed_conversations"])
                    if row["status"] not in {"ready", "failed", "building"}:
                        raise ValueError("search_status_invalid")
                    if row["status"] == "building":
                        builder.blockers.append(
                            _ContentBlocker(
                                table, original, "native_search_build_unresolved"
                            )
                        )
                    for key in (
                        "created_at",
                        "updated_at",
                        "completed_at",
                        "lease_expires_at",
                        "error_code",
                    ):
                        _text(
                            row[key],
                            optional=key
                            in {"completed_at", "lease_expires_at", "error_code"},
                        )
                elif table.endswith("_documents"):
                    _integer(row["document_id"], 1)
                    _integer(row["character_id"], 1)
                    for key in (
                        "conversation_id",
                        "character_label",
                        "title",
                        "body",
                        "eligibility_digest",
                        "validated_eligibility_digest",
                        "generation_id",
                    ):
                        _text(row[key])
                if "updated_at" in row:
                    _text(row["updated_at"], limit=80)
                if "enqueued_at" in row:
                    _text(row["enqueued_at"], limit=80)
            except (ValueError, TypeError):
                builder.blockers.append(
                    _ContentBlocker(table, original, "native_search_state_invalid")
                )
            provenance.append(
                _OperationalProvenance(
                    table,
                    original,
                    _sqlite_row_digest(
                        tuple(row[column] for column in builder.specs[table].columns)
                    ),
                    builder.identities.source_manifest_digest,
                    "retain_destination_authority_native_derived",
                )
            )
    return tuple(provenance)


def _all_originals(actors):
    settings = actors.workflows.settings
    return tuple(
        {
            (r.table, r.original): r
            for r in (
                *settings.graph.source_rows,
                *settings.source_rows,
                *actors.workflows.source_rows,
                *actors.source_rows,
            )
        }.values()
    )


def _ownership(builder, actors):
    from .mobile_content_canvas import _CANVAS_TABLES
    from .mobile_content_core import _CORE_TABLES, _MESSAGE_PENDING
    from .mobile_content_rules import _RULE_TABLES

    fields = {}
    for table in _CORE_TABLES + _CANVAS_TABLES + _RULE_TABLES:
        if table == "conversations":
            continue
        fields[table] = tuple(
            c
            for c in builder.specs[table].columns
            if table != "messages" or c not in _MESSAGE_PENDING
        )
    for table, columns in actors.field_ownership + tuple(
        (t, builder.specs[t].columns) for t in _TABLES
    ):
        if set(fields.get(table, ())) & set(columns):
            raise ValueError("mobile_c_field_ownership_duplicate")
        fields[table] = fields.get(table, ()) + columns
    expected = {s.table: s.columns for s in builder.specs.values() if s.handler == "C"}
    if set(fields) != set(expected) or any(
        set(fields[t]) != set(expected[t]) for t in fields
    ):
        raise ValueError("mobile_c_field_ownership_incomplete")
    return tuple((t, expected[t]) for t in fields)


def _plan_content(reads):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_content_native_snapshot_required")
    reads.check()
    actors = reads.content_actor_plan()
    builder = _GraphBuilder(reads, actors.identities, _TABLES)
    ownership = _ownership(builder, actors)
    builder.rows.extend(actors.rows)
    builder.blockers.extend(actors.blockers)
    _context_and_accounting(builder)
    provenance = _search(builder)
    _memory_rows(builder, _all_originals(actors))
    blocked = {(b.table, b.original) for b in builder.blockers}
    builder.rows = [r for r in builder.rows if (r.table, r.original) not in blocked]
    _close_memory_dependencies(builder)
    successors, discharged = _legacy_summary(builder, actors)
    blocked = {(b.table, b.original) for b in builder.blockers}
    builder.rows = [r for r in builder.rows if (r.table, r.original) not in blocked]
    builder.close_dependencies()
    accepted = {(r.table, r.original) for r in builder.rows}
    for row in builder.rows:
        if set(dict(row.values)) | set(row.pending_columns) != set(
            builder.specs[row.table].columns
        ):
            raise ValueError("mobile_c_row_incomplete")
    reads.check()
    return _ContentPlan(
        actors,
        tuple(builder.rows),
        _all_originals(actors) + builder.retained(),
        tuple(builder.blockers),
        provenance,
        tuple(s for s in successors if (s.table, s.original) in accepted),
        tuple(s for s in discharged if (s.table, s.original) in accepted),
        ownership,
    )


def _memory_rows(builder, originals):
    """Prove stored memory against the captured native bodies before mapping."""
    from tldw_chatbook.Chat import console_context_repository as native
    from tldw_chatbook.Chat.console_context_compaction import (
        DurableConversationUnit,
        DurableMessageSnapshot,
        _sealed_memory_marker,
        complete_durable_units,
        prefix_digest,
    )

    from .mobile_content_graph import _UnprovedReference
    from .mobile_content_settings import _json

    source_messages = {
        r.original[0]: dict(r.values) for r in originals if r.table == "messages"
    }
    mapped_messages = {
        r.original[0]: dict(r.values) for r in builder.rows if r.table == "messages"
    }
    source_attachments = [
        dict(r.values) for r in originals if r.table == "message_attachments"
    ]
    accepted_attachments = {
        r.original for r in builder.rows if r.table == "message_attachments"
    }
    source_memories = {
        r["id"]: r for r in builder.source["console_conversation_memories"]
    }
    source_scopes = {
        r["memory_id"]: r for r in builder.source["console_conversation_memory_scopes"]
    }

    def message(identity, conversation):
        row = source_messages.get(identity)
        if row is None or identity not in mapped_messages:
            raise _UnprovedReference("message", identity)
        if row["conversation_id"] != conversation:
            raise ValueError("memory_message_owner_mismatch")
        builder.ref("message", identity)
        return row

    def state(identity, conversation, *, mapped):
        original = message(identity, conversation)
        row = mapped_messages[identity] if mapped else original
        root = original["variant_of"] or identity
        variants = [
            r
            for r in source_messages.values()
            if r["id"] == root or r["variant_of"] == root
        ]
        for candidate in variants:
            message(candidate["id"], conversation)
        variants.sort(
            key=lambda r: (
                r["variant_number"],
                mapped_messages[r["id"]]["id"] if mapped else r["id"],
            )
        )
        live = [r for r in variants if not r["deleted"]]
        selected_id = selected_index = None
        content_row = row
        owner = identity
        if len(live) > 1:
            selected = [
                (index, r)
                for index, r in enumerate(live)
                if r["is_selected_variant"] == 1
            ]
            if len(selected) != 1:
                raise ValueError("memory_variant_ambiguous")
            selected_index, selected_source = selected[0]
            owner = selected_source["id"]
            content_row = mapped_messages[owner] if mapped else selected_source
            selected_id = content_row["id"]
        digests = []
        if content_row["image_data"] is not None:
            digests.append(
                native.persisted_attachment_digest(
                    position=0,
                    mime_type=str(content_row["image_mime_type"] or ""),
                    display_name="",
                    data=bytes(content_row["image_data"]),
                )
            )
        for attachment in sorted(
            (a for a in source_attachments if a["message_id"] == owner),
            key=lambda a: a["position"],
        ):
            if (owner, attachment["position"]) not in accepted_attachments:
                raise _UnprovedReference(
                    "message_attachment", (owner, attachment["position"])
                )
            digests.append(
                native.persisted_attachment_digest(
                    position=attachment["position"],
                    mime_type=attachment["mime_type"],
                    display_name=attachment["display_name"] or "",
                    data=attachment["data"],
                )
            )
        return native._PersistedLineageState(
            native.PersistedLineageFenceRow(
                row["id"],
                row["parent_message_id"],
                row["version"],
                bool(row["deleted"]),
                native._digest_json(content_row["content"]),
                selected_id,
                selected_index,
                tuple(digests),
            ),
            str(content_row["role"] or content_row["sender"]),
            content_row["content"],
        )

    def snapshots(states):
        return tuple(
            DurableMessageSnapshot(
                message_id=s.fence.message_id,
                version=s.fence.version,
                role=s.role,
                content=s.content,
                parent_message_id=s.fence.parent_message_id,
                deleted=s.fence.deleted,
                selected_variant_id=s.fence.selected_variant_id,
                selected_variant_index=s.fence.selected_variant_index,
                attachment_digests=s.fence.attachment_digests,
            )
            for s in states
        )

    def projected(snapshot, original_id):
        """The native manual-memory checkpoint's fixed virtual row grammar."""
        from tldw_chatbook.Chat.provider_continuation import (
            parse_provider_continuation_json,
        )

        raw = source_messages[original_id]["provider_continuation_json"]
        if raw is None:
            return (snapshot,)
        checkpoint = parse_provider_continuation_json(raw)
        if checkpoint.state != "complete":
            raise _UnprovedReference("memory_checkpoint", original_id)
        result, parent = [], snapshot.parent_message_id
        for index, round_ in enumerate(checkpoint.rounds):
            if not round_.calls:
                continue
            identity = f"{snapshot.message_id}:tool-calls:{index}"
            result.append(
                DurableMessageSnapshot(
                    identity,
                    snapshot.version,
                    "assistant",
                    round_.assistant_content,
                    parent_message_id=parent,
                    tool_calls=tuple(
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                        for call in round_.calls
                    ),
                )
            )
            parent = identity
            for call_index, call in enumerate(round_.calls):
                if call.result is None:
                    raise _UnprovedReference("memory_checkpoint", original_id)
                identity = f"{snapshot.message_id}:tool-result:{index}:{call_index}"
                result.append(
                    DurableMessageSnapshot(
                        identity,
                        snapshot.version,
                        "tool",
                        call.result.value,
                        parent_message_id=parent,
                        tool_call_id=call.call_id,
                    )
                )
                parent = identity
        return tuple(result) + (replace(snapshot, parent_message_id=parent),)

    def memory(row):
        for key in ("prompt_revision", "revision"):
            _integer(row[key], 1)
        for key in ("input_tokens", "output_tokens", "before_tokens", "after_tokens"):
            _integer(row[key])
        for key in (
            "id",
            "conversation_id",
            "boundary_message_id",
            "captured_leaf_message_id",
            "lineage_json",
            "summary_text",
            "provider",
            "model",
            "prompt_id",
            "prompt_digest",
            "selected_units_json",
            "summarized_prefix_digest",
            "created_at",
            "source_kind",
        ):
            _text(row[key])
        native._memory_from_row(row)
        for key in ("active",):
            _integer(row[key], 0, 1)
        _text(row["reset_at"], optional=True, limit=80)
        lineage = _object('{"items":' + row["lineage_json"] + "}")["items"]
        if (
            not lineage
            or any(type(mid) is not str for mid in lineage)
            or len(set(lineage)) != len(lineage)
        ):
            raise ValueError("memory_lineage_invalid")
        conversation = row["conversation_id"]
        before = tuple(state(mid, conversation, mapped=False) for mid in lineage)
        after = tuple(state(mid, conversation, mapped=True) for mid in lineage)
        if (
            before[0].fence.parent_message_id is not None
            or before[-1].fence.message_id != row["captured_leaf_message_id"]
        ):
            raise ValueError("memory_lineage_boundary_invalid")
        if any(
            child.fence.parent_message_id != parent.fence.message_id
            for parent, child in pairwise(before)
        ):
            raise ValueError("memory_lineage_not_chain")
        if row["boundary_message_id"] not in lineage:
            raise ValueError("memory_boundary_missing")
        boundary = lineage.index(row["boundary_message_id"]) + 1
        raw_snapshots, mapped_snapshots = snapshots(before), snapshots(after)
        candidates = {}
        for position, (original_id, raw, mapped) in enumerate(
            zip(
                lineage[:boundary],
                raw_snapshots[:boundary],
                mapped_snapshots[:boundary],
                strict=True,
            )
        ):
            pairs = [(raw, mapped)]
            if source_messages[original_id]["provider_continuation_json"] is not None:
                raw_projection = projected(raw, original_id)
                mapped_projection = projected(mapped, original_id)
                pairs.extend(zip(raw_projection, mapped_projection, strict=True))
            for index, (old, new) in enumerate(pairs):
                if old.message_id != original_id and old.message_id in source_messages:
                    raise _UnprovedReference("memory_snapshot_identity", old.message_id)
                rank = (
                    position,
                    len(pairs) if old.message_id == original_id else index,
                )
                candidates.setdefault(old.message_id, []).append((old, new, rank))
        selected = _object('{"items":' + row["selected_units_json"] + "}")["items"]
        transformed = []
        frontier = (-1, -1)
        tails = []
        for unit in selected:
            if type(unit) is not dict:
                raise ValueError("memory_unit_invalid")
            if set(unit) == {"messages"}:
                if type(unit["messages"]) is not list or not unit["messages"]:
                    raise ValueError("memory_unit_empty")
                if tails:
                    raise ValueError("memory_unit_after_marker")
                old, new = [], []
                for proof in unit["messages"]:
                    if (
                        type(proof) is not dict
                        or proof.get("message_id") not in candidates
                    ):
                        raise _UnprovedReference("memory_snapshot", row["id"])
                    match = next(
                        (
                            (a, b, rank)
                            for a, b, rank in candidates[proof["message_id"]]
                            if proof == a.provenance_payload()
                        ),
                        None,
                    )
                    if match is None:
                        raise _UnprovedReference("memory_snapshot", row["id"])
                    a, b, rank = match
                    if rank <= frontier or rank[0] >= boundary:
                        raise ValueError("memory_unit_order_invalid")
                    frontier = rank
                    old.append(a)
                    new.append(b)
                if complete_durable_units(old) != (
                    DurableConversationUnit(tuple(old)),
                ):
                    raise ValueError("memory_unit_incomplete")
                transformed.append(
                    DurableConversationUnit(tuple(new)).provenance_payload()
                )
            elif unit.get("kind") == "sealed_prior_memory":
                _shape(
                    unit,
                    (
                        "kind",
                        "memory_id",
                        "memory_revision",
                        "start_message_id",
                        "end_message_id",
                    ),
                )
                previous = source_memories.get(unit["memory_id"])
                scope = source_scopes.get(unit["memory_id"])
                if (
                    previous is None
                    or scope is None
                    or previous["conversation_id"] != conversation
                    or previous["id"] == row["id"]
                ):
                    raise _UnprovedReference("memory", unit["memory_id"])
                if unit != _sealed_memory_marker(
                    native._memory_from_row(previous),
                    native._memory_scope_from_row(scope),
                ):
                    raise _UnprovedReference(
                        "memory_revision", (unit["memory_id"], unit["memory_revision"])
                    )
                if (
                    tails
                    or scope["coverage_kind"] != "range"
                    or unit["start_message_id"] not in lineage
                    or unit["end_message_id"] not in lineage
                ):
                    raise ValueError("memory_marker_scope_invalid")
                start, end = (
                    lineage.index(unit["start_message_id"]),
                    lineage.index(unit["end_message_id"]),
                )
                if start <= frontier[0] or not start < end < boundary:
                    raise ValueError("memory_marker_order_invalid")
                frontier = (end, len(candidates) + 1)
                transformed.append(
                    unit
                    | {
                        "memory_id": builder.ref("memory", unit["memory_id"]),
                        "start_message_id": builder.ref(
                            "message", unit["start_message_id"], optional=True
                        ),
                        "end_message_id": builder.ref(
                            "message", unit["end_message_id"]
                        ),
                    }
                )
            elif unit.get("kind") == "focus_topic":
                _shape(unit, ("kind", "topic", "applied"))
                if tails:
                    raise ValueError("memory_focus_order_invalid")
                tails.append("focus_topic")
                _text(unit["topic"])
                if type(unit["applied"]) is not bool:
                    raise ValueError("memory_focus_invalid")
                transformed.append(unit)
            elif unit == {"kind": "compaction_engine", "engine": "provider_native"}:
                if "compaction_engine" in tails:
                    raise ValueError("memory_engine_duplicate")
                tails.append("compaction_engine")
                transformed.append(unit)
            else:
                raise ValueError("memory_unit_unknown")
        if selected and frontier[0] != boundary - 1:
            raise ValueError("memory_units_boundary_invalid")
        if row["summarized_prefix_digest"] not in {
            native._prefix_digest_from_persisted(before[:boundary]),
            prefix_digest(raw_snapshots[:boundary]),
        }:
            raise _UnprovedReference("memory_prefix", row["id"])
        if row["source_kind"] == "generated" and row["id"] not in source_scopes:
            raise _UnprovedReference("memory_scope", row["id"])
        values = row | {
            "id": builder.ref("memory", row["id"]),
            "conversation_id": builder.ref("conversation", conversation),
            "boundary_message_id": builder.ref("message", row["boundary_message_id"]),
            "captured_leaf_message_id": builder.ref(
                "message", row["captured_leaf_message_id"]
            ),
            "lineage_json": _json([s.fence.message_id for s in after]),
            "selected_units_json": row["selected_units_json"]
            if selected == transformed
            else _json(transformed),
            "summarized_prefix_digest": native._prefix_digest_from_persisted(
                after[:boundary]
            ),
        }
        native._memory_from_row(values)
        builder.emit("console_conversation_memories", row, values)

    builder.handle("console_conversation_memories", memory)

    def scope(row):
        native._memory_scope_from_row(row)
        original = source_memories.get(row["memory_id"])
        if original is None:
            raise _UnprovedReference("memory", row["memory_id"])
        if original["conversation_id"] != row["conversation_id"]:
            raise ValueError("memory_scope_owner_mismatch")
        lineage = json.loads(original["lineage_json"])
        anchor = row["selection_anchor_message_id"]
        if anchor is not None:
            if (
                anchor not in lineage
                or state(anchor, row["conversation_id"], mapped=False).role != "user"
            ):
                raise ValueError("memory_scope_anchor_invalid")
            boundary = lineage.index(original["boundary_message_id"])
            position = lineage.index(anchor)
            if not (
                boundary < position
                if row["coverage_kind"] == "prefix"
                else position < boundary
            ):
                raise ValueError("memory_scope_order_invalid")
        values = row | {
            "memory_id": builder.ref("memory", row["memory_id"]),
            "conversation_id": builder.ref("conversation", row["conversation_id"]),
            "selection_anchor_message_id": builder.ref(
                "message", anchor, optional=True
            ),
        }
        native._memory_scope_from_row(values)
        builder.emit("console_conversation_memory_scopes", row, values)

    builder.handle("console_conversation_memory_scopes", scope)

    def selection(row):
        for key in ("suppresses_legacy", "active"):
            _integer(row[key], 0, 1)
        native._memory_selection_from_row(row)
        message(row["activation_message_id"], row["conversation_id"])
        if row["selected_memory_id"] is not None:
            memory = source_memories.get(row["selected_memory_id"])
            if memory is None:
                raise _UnprovedReference("memory", row["selected_memory_id"])
            if (
                memory["conversation_id"] != row["conversation_id"]
                or memory["captured_leaf_message_id"] != row["activation_message_id"]
            ):
                raise ValueError("memory_selection_owner_mismatch")
        values = row | {
            "sequence": builder.ref("memory_sequence", row["sequence"]),
            "selection_id": builder.ref("memory_selection", row["selection_id"]),
            "conversation_id": builder.ref("conversation", row["conversation_id"]),
            "activation_message_id": builder.ref(
                "message", row["activation_message_id"]
            ),
            "selected_memory_id": builder.ref(
                "memory", row["selected_memory_id"], optional=True
            ),
        }
        native._memory_selection_from_row(values)
        builder.emit("console_conversation_memory_selections", row, values)

    builder.handle("console_conversation_memory_selections", selection)


def _close_memory_dependencies(builder):
    """A memory and its scope form one native unit; failed proof closes dependents."""
    memory_table = "console_conversation_memories"
    scope_table = "console_conversation_memory_scopes"
    scopes = {r.original[0] for r in builder.rows if r.table == scope_table}
    blocked_scopes = {b.original[0] for b in builder.blockers if b.table == scope_table}
    memories = {r.original[0]: r for r in builder.rows if r.table == memory_table}
    edges = {
        key: {
            identity
            for domain, identity in row.dependencies
            if domain == "memory" and identity != key
        }
        for key, row in memories.items()
    }
    for identity, row in memories.items():
        missing_scope = dict(row.values)["source_kind"] == "generated" and (
            identity not in scopes or identity in blocked_scopes
        )
        frontier, seen = list(edges[identity]), set()
        while frontier:
            target = frontier.pop()
            if target == identity:
                builder.blockers.append(
                    _ContentBlocker(
                        memory_table, row.original, "native_memory_provenance_cycle"
                    )
                )
                break
            if target not in seen:
                seen.add(target)
                frontier.extend(edges.get(target, ()))
        if missing_scope:
            builder.blockers.append(
                _ContentBlocker(
                    memory_table,
                    row.original,
                    "dependency_blocked",
                    "memory_scope",
                    identity,
                )
            )
    blocked = {(b.table, b.original) for b in builder.blockers}
    builder.rows = [r for r in builder.rows if (r.table, r.original) not in blocked]
    builder.close_dependencies()


def _legacy_summary(builder, actors):
    """Discharge only C2's exact legacy summary/boundary association."""
    originals = {
        (r.table, r.original): dict(r.values)
        for r in actors.workflows.settings.source_rows
    }
    projected = {(r.table, r.original): dict(r.values) for r in builder.rows}
    messages = {
        r.original[0]: dict(r.values)
        for r in actors.workflows.settings.source_rows
        if r.table == "messages"
    }
    successors, discharged = [], []
    for successor in actors.successors:
        if successor.handler != "C3":
            successors.append(successor)
            continue
        key = successor.table, successor.original
        original, mapped = originals.get(key), projected.get(key)
        try:
            if (
                successor.table != "conversations"
                or successor.domain != "conversation_memory"
                or original is None
                or mapped is None
            ):
                raise ValueError("memory_successor_unknown")
            if original["context_summary"] != mapped["context_summary"]:
                raise ValueError("legacy_summary_changed")
            _text(original["context_summary"], optional=True)
            boundary = original["summary_boundary_message_id"]
            if boundary is not None and (
                boundary not in messages
                or messages[boundary]["conversation_id"] != original["id"]
                or builder.identities.mapped("message", boundary)
                != mapped["summary_boundary_message_id"]
            ):
                raise ValueError("legacy_boundary_invalid")
        except (ValueError, KeyError, TypeError):
            builder.blockers.append(
                _ContentBlocker(
                    successor.table, successor.original, "native_legacy_memory_invalid"
                )
            )
        else:
            discharged.append(successor)
    closed = {(s.table, s.original, "C3:" + s.domain) for s in discharged}
    builder.rows = [
        replace(
            row,
            required_proofs=tuple(
                p
                for p in row.required_proofs
                if (row.table, row.original, p) not in closed
            ),
        )
        for row in builder.rows
    ]
    return tuple(successors), tuple(discharged)
