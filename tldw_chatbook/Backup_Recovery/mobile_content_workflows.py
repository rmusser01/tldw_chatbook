"""Fixed native workflow projections; historical data grants no live authority."""

import json
import math
import re
from dataclasses import dataclass, field, replace

from .mobile_content_graph import (
    _ContentBlocker,
    _ContentRow,
    _GraphBuilder,
    _NativeConflict,
    _UnprovedReference,
)
from .mobile_content_settings import _json, _SettingsPlan, _SettingsSuccessor
from .mobile_workspace_plan import _object

_TABLES = (
    "console_dispatch_checkpoints",
    "console_hook_continuation_receipts",
    "console_machine_followup_receipts",
    "fleet_progress_messages",
    "message_generation_metadata",
    "message_trajectory_metadata",
    "transcript_annotations",
)

_RETRIEVAL_KINDS = frozenset(
    {
        "retrieval_started",
        "retrieval_failed",
        "retrieval_candidates_selected",
        "retrieval_completed",
        "context_attached",
        "context_injected",
    }
)


@dataclass(frozen=True)
class _WorkflowProvenance:
    table: str
    original: tuple[str | int, ...]
    source_identity: str
    projected_identity: str
    meaning: str


@dataclass(frozen=True)
class _WorkflowPlan:
    settings: _SettingsPlan = field(repr=False)
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    successors: tuple[_SettingsSuccessor, ...]
    discharged: tuple[_SettingsSuccessor, ...]
    provenance: tuple[_WorkflowProvenance, ...]
    pending_handlers: tuple[str, ...] = ("C2c", "C3", "T", "S", "A-runs")

    @property
    def identities(self):
        return self.settings.identities

    def require_complete(self) -> None:
        raise ValueError("mobile_content_handlers_pending")


def _text(value, *, optional=False, limit=None):
    if value is None and optional:
        return value
    if type(value) is not str or (
        limit is not None and (not value or len(value) > limit)
    ):
        raise ValueError("invalid_workflow_text")
    return value


def _integer(value, minimum=0, maximum=2**63 - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid_workflow_integer")
    return value


def _shape(value, required, optional=()):
    if (
        type(value) is not dict
        or set(value) - set(required) - set(optional)
        or set(required) - set(value)
    ):
        raise ValueError("unknown_workflow_payload")


def _event_ref(builder, value):
    if value is None:
        return None
    _text(value, limit=512)
    if value.startswith("message:"):
        return "message:" + builder.ref("message", value[len("message:") :])
    if value.startswith("console-trace:"):
        identity, separator, kind = value[len("console-trace:") :].rpartition(
            ":retrieval:"
        )
        if separator and kind in _RETRIEVAL_KINDS:
            return (
                "console-trace:"
                + builder.ref("message", identity)
                + ":retrieval:"
                + kind
            )
    # Other producer identities are not message IDs merely because they look alike.
    raise _UnprovedReference("trajectory_event", value)


def _trajectory_payload(builder, row, successor):
    kind, raw = row["event_kind"], row["payload_json"]
    payload = None if raw is None else _object(raw)
    if kind in {"user", "assistant"}:
        if payload is None:
            return raw
        if kind != "assistant":
            raise ValueError("unknown_user_payload")
        _shape(payload, ("trace_version", "model_status"))
        if type(payload["trace_version"]) is not int or payload["trace_version"] != 2:
            raise ValueError("unknown_trace_version")
        _text(payload["model_status"], optional=True)
    elif kind in {"tool_call", "tool_result"}:
        _shape(
            payload,
            ("name", "args", "result"),
            ("field_states", "sensitivity", "truncated"),
        )
        _text(payload["name"])
        if payload["args"] is not None or not isinstance(
            payload["result"], (str, type(None))
        ):
            raise ValueError("unknown_tool_payload")
        if "field_states" in payload:
            _shape(payload["field_states"], ("args", "result"))
            if payload["field_states"]["args"] != "not_available" or payload[
                "field_states"
            ]["result"] not in {
                "not_available",
                "omitted",
                "redacted",
                "observed",
                "truncated",
            }:
                raise ValueError("unknown_tool_field_state")
        if "truncated" in payload and type(payload["truncated"]) is not bool:
            raise ValueError("invalid_truncation")
        if "sensitivity" in payload and payload["sensitivity"] not in {
            "path",
            "tool_content",
        }:
            raise ValueError("unknown_tool_sensitivity")
    elif kind == "user_feedback":
        _shape(payload, ("action", "quote"), ("comment",))
        for value in payload.values():
            _text(value)
    elif kind == "library_preparation":
        from tldw_chatbook.Chat.library_preparation import (
            decode_library_preparation_event,
        )

        decode_library_preparation_event(raw)
    elif kind == "library_activity":
        from tldw_chatbook.Chat.library_activity import (
            decode_library_activity_event,
            encode_library_activity_event,
        )

        event = decode_library_activity_event(raw)
        refs = []
        for ref in event.source_refs:
            domain = {
                "note": "note",
                "notes": "note",
                "conversation": "conversation",
                "conversations": "conversation",
            }.get(ref.source_type)
            if domain is None:
                raise _UnprovedReference(
                    "library_source:" + ref.source_type, ref.source_id
                )
            refs.append(replace(ref, source_id=builder.ref(domain, ref.source_id)))
        run = builder.ref("run", event.run_id)
        parent = builder.ref("run", event.parent_run_id, optional=True)
        successor("A-runs", "run", event.run_id, run)
        if parent is not None:
            successor("A-runs", "run", event.parent_run_id, parent)
        return encode_library_activity_event(
            replace(event, run_id=run, parent_run_id=parent, source_refs=tuple(refs))
        )
    elif kind == "capture_failed":
        _shape(
            payload, ("event_id", "summary", "status", "field_states", "sensitivity")
        )
        if (
            not re.fullmatch(r"capture-failed:[0-9a-f]{64}", payload["event_id"])
            or payload["summary"] != "Trace capture failed"
            or payload["status"] != "incomplete"
            or payload["field_states"] != {"payload": "capture_failed"}
            or payload["sensitivity"] != "diagnostic"
        ):
            raise ValueError("invalid_capture_failure")
        # The original failed write's source_events are absent: never mint a new hash.
    elif kind in _RETRIEVAL_KINDS | {
        "context_attached",
        "context_injected",
        "message_regenerated",
        "message_edited",
        "branch_selected",
        "model_retry",
        "message_retry_requested",
        "model_cancelled",
        "model_error",
    }:
        _shape(
            payload,
            (
                "summary",
                "status",
                "event_id",
                "parent_event_id",
                "source_event_id",
                "replacement_event_id",
                "field_states",
                "sensitivity",
            ),
        )
        for name in ("summary", "status", "sensitivity"):
            _text(payload[name])
        if type(payload["field_states"]) is not dict or any(
            type(k) is not str or type(v) is not str
            for k, v in payload["field_states"].items()
        ):
            raise ValueError("invalid_field_states")
        value = dict(payload)
        for name in (
            "event_id",
            "parent_event_id",
            "source_event_id",
            "replacement_event_id",
        ):
            value[name] = _event_ref(builder, value[name])
        return raw if value == payload else _json(value)
    else:
        raise _UnprovedReference("trajectory_event_kind", kind)
    return raw


def _hook_metadata_join(row, messages):
    """Require the actual native receipt's USER/assistant/parent audit relation."""
    child = messages[row["assistant_message_id"]]
    user = messages[child["parent_message_id"]]
    if (
        child["role"] != "assistant"
        or user["role"] != "user"
        or child["conversation_id"] != row["conversation_id"]
        or user["conversation_id"] != row["conversation_id"]
        or user["parent_message_id"] != row["parent_assistant_message_id"]
        or _object(user["metadata_json"])
        != {
            "origin": "hook",
            "initiator": "hook_continuation",
            "parent_turn_id": row["parent_turn_id"],
            "stop_event_id": row["stop_event_id"],
        }
    ):
        raise ValueError("invalid_hook_message_relation")


def _checkpoint_join(row, conversations, messages):
    conversation = conversations[row["conversation_id"]]
    user, assistant = (
        messages[row["user_message_id"]],
        messages[row["assistant_message_id"]],
    )
    return row | {
        "conversation_deleted": conversation["deleted"],
        "user_conversation_id": user["conversation_id"],
        "user_role": user["role"],
        "user_metadata_json": user["metadata_json"],
        "current_user_version": user["version"],
        "user_deleted": user["deleted"],
        "assistant_conversation_id": assistant["conversation_id"],
        "assistant_role": assistant["role"],
        "current_assistant_version": assistant["version"],
        "assistant_deleted": assistant["deleted"],
        "assistant_state": assistant["assistant_generation_state"],
        "provider_continuation_json": assistant["provider_continuation_json"],
        "thinking_blocks_json": assistant["thinking_blocks_json"],
    }


def _plan_content_workflows(reads):
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        dump_console_turn_library_authority_json,
    )
    from tldw_chatbook.Chat.console_dispatch_repository import ConsoleDispatchRepository
    from tldw_chatbook.Chat.provider_continuation import (
        parse_provider_continuation_json,
        read_provider_continuation_json,
    )

    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_content_native_snapshot_required")
    reads.check()
    settings = reads.content_settings_plan()
    builder = _GraphBuilder(reads, settings.identities, _TABLES)
    builder.blockers.extend(settings.blockers)
    builder.rows.extend(
        r for r in settings.graph.rows if r.table not in {"conversations", "messages"}
    )
    builder.rows.extend(settings.rows)
    originals = {
        table: {
            r.original[0]: dict(r.values)
            for r in settings.source_rows
            if r.table == table
        }
        for table in ("conversations", "messages")
    }
    projected = {
        table: {
            dict(r.values)["id"]: dict(r.values)
            for r in settings.rows
            if r.table == table
        }
        for table in originals
    }
    successors, discharged, provenance = [], [], []

    def require(table, row, handler, domain, source, mapped):
        successors.append(
            _SettingsSuccessor(
                table, builder.original(table, row), handler, domain, source, mapped
            )
        )

    def message_ref(row, column="message_id", *, optional=False):
        identity = row[column]
        mapped = builder.ref("message", identity, optional=optional)
        if (
            identity is not None
            and originals["messages"][identity]["conversation_id"]
            != row["conversation_id"]
        ):
            raise ValueError("workflow_message_conversation_mismatch")
        return mapped

    def active_path(conversation_id):
        # Mirrors the native recursive active-path relation over captured C1 rows.
        conversation = originals["conversations"][conversation_id]
        current = conversation["active_leaf_message_id"]
        path = set()
        while (
            current is not None and current not in path and not conversation["deleted"]
        ):
            message = originals["messages"].get(current)
            if message is None or message["conversation_id"] != conversation_id:
                break
            path.add(current)
            current = message["parent_message_id"]
        return path

    def checkpoint(row):
        joined = _checkpoint_join(
            row, originals["conversations"], originals["messages"]
        )
        native, error = ConsoleDispatchRepository._checkpoint_from_row(joined)
        on_path = row["assistant_message_id"] in active_path(row["conversation_id"])
        if error is not None or native is None or not on_path:
            reason = "native_checkpoint_owner_conflict"
            if not on_path and native is not None:
                # The native stranded-owner branch must either repair its pointer
                # or quarantine it. Neither operation is an import projection.
                reason = "native_checkpoint_branch_conflict"
            elif error != "invalid_checkpoint_owner" or row["schema_version"] != 1:
                raise ValueError("invalid_native_checkpoint")
            elif on_path and ConsoleDispatchRepository._valid_reconcile_pair(joined):
                continuation = None
                if joined["provider_continuation_json"] is not None:
                    continuation = parse_provider_continuation_json(
                        joined["provider_continuation_json"]
                    )
                if (continuation is not None and continuation.state == "active") or (
                    continuation is None
                    and joined["assistant_state"]
                    in {"complete", "stopped", "failed", "discarded"}
                ):
                    reason = "native_checkpoint_reconciliation_required"
            builder.blockers.append(
                _ContentBlocker(_TABLES[0], builder.original(_TABLES[0], row), reason)
            )
            return
        for name in ("created_at", "updated_at"):
            _text(row[name])
        for name in (
            "schema_version",
            "checkpoint_revision",
            "user_message_version",
            "assistant_message_version",
        ):
            _integer(row[name], 1)
        scope = native.frozen_authority.scope_snapshot
        if scope.media_ids:
            raise _UnprovedReference("media", scope.media_ids[0])
        authority = replace(
            native.frozen_authority,
            scope_snapshot=replace(
                scope,
                note_ids=tuple(builder.ref("note", value) for value in scope.note_ids),
            ),
        )
        values = row | {
            "conversation_id": builder.ref("conversation", row["conversation_id"]),
            "assistant_message_id": message_ref(row, "assistant_message_id"),
            "user_message_id": message_ref(row, "user_message_id"),
            "frozen_authority_json": dump_console_turn_library_authority_json(
                authority
            ),
        }
        if row["agent_chat_start_attempt_id"] is not None:
            identity = row["agent_chat_start_attempt_id"]
            values["agent_chat_start_attempt_id"] = builder.ref(
                "chat_start_attempt", identity
            )
            require(
                _TABLES[0],
                row,
                "A-runs",
                "chat_start_attempt",
                identity,
                values["agent_chat_start_attempt_id"],
            )
        mapped, error = ConsoleDispatchRepository._checkpoint_from_row(
            _checkpoint_join(values, projected["conversations"], projected["messages"])
        )
        if error is not None or mapped is None:
            raise ValueError("invalid_mapped_checkpoint")
        if any(
            r["conversation_id"] == values["conversation_id"]
            or r["preparation_id"] == values["preparation_id"]
            or (
                values["agent_chat_start_attempt_id"] is not None
                and r["agent_chat_start_attempt_id"]
                == values["agent_chat_start_attempt_id"]
            )
            for r in builder.canonical[_TABLES[0]]
        ):
            raise _NativeConflict()
        builder.emit(
            _TABLES[0], row, values, proofs=("S:imported_dispatch_quarantine",)
        )

    builder.handle(_TABLES[0], checkpoint)

    def receipt(row, *, machine):
        table = _TABLES[2 if machine else 1]
        for name in (
            ("operation_id", "settlement_id") if machine else ("stop_event_id",)
        ) + ("parent_turn_id", "chain_id"):
            _text(row[name], limit=256)
        _integer(row["admitted_turns"], 1, 3)
        _text(row["parent_assistant_message_id"], limit=256)
        if any(
            builder.original(table, r) == builder.original(table, row)
            for r in builder.canonical[table]
        ):
            raise _NativeConflict()
        if machine:
            _integer(row["native_turns"], 0, min(2, row["admitted_turns"]))
            # Stored receipts intentionally omit source_message_version; no current
            # message version can prove the version at historical acceptance.
            contributors = json.loads(row["contributors_json"])
            if (
                contributors not in [["native"], ["hook"], ["native", "hook"]]
                or len(row["contributors_json"]) > 64
            ):
                raise ValueError("invalid_machine_contributors")
            if "hook" in contributors and not any(
                builder.original(_TABLES[1], h)
                in {r.original for r in builder.rows if r.table == _TABLES[1]}
                and h["parent_turn_id"] == row["parent_turn_id"]
                and h["conversation_id"] == row["conversation_id"]
                and h["assistant_message_id"] == row["assistant_message_id"]
                and h["parent_assistant_message_id"]
                == row["parent_assistant_message_id"]
                and h["chain_id"] == row["chain_id"]
                and h["admitted_turns"] == row["admitted_turns"]
                for h in builder.source[_TABLES[1]]
            ):
                raise ValueError("missing_genuine_hook_receipt")
        elif row["initiator"] != "hook_continuation":
            raise ValueError("invalid_hook_initiator")
        if not machine:
            _hook_metadata_join(row, originals["messages"])
        parent = originals["messages"][row["parent_assistant_message_id"]]
        child = originals["messages"][row["assistant_message_id"]]
        if parent["role"] != "assistant" or child["role"] != "assistant":
            raise ValueError("invalid_receipt_parent")
        values = row | {
            "conversation_id": builder.ref("conversation", row["conversation_id"]),
            "parent_assistant_message_id": message_ref(
                row, "parent_assistant_message_id"
            ),
            "assistant_message_id": message_ref(row, "assistant_message_id"),
        }
        if not machine:
            _hook_metadata_join(values, projected["messages"])
        builder.emit(table, row, values, proofs=("S:imported_followup_quarantine",))

    builder.handle(_TABLES[1], lambda row: receipt(row, machine=False))
    builder.handle(_TABLES[2], lambda row: receipt(row, machine=True))

    def fleet(row):
        from tldw_chatbook.Agents.fleet_messages import (
            MAX_AGENT_CHARS,
            MAX_IDENTITY_CHARS,
            MAX_INBOX_PENDING,
            MAX_INBOX_PENDING_CHARS,
            MAX_MESSAGE_CHARS,
            _validate_text,
        )

        for name in ("message_id", "handle_id", "run_id", "parent_run_id"):
            _validate_text(row[name], MAX_IDENTITY_CHARS)
        if row["chain_id"] is not None:
            _validate_text(row["chain_id"], MAX_IDENTITY_CHARS)
        _validate_text(row["agent"], MAX_AGENT_CHARS)
        _validate_text(row["body"], MAX_MESSAGE_CHARS)
        _text(row["created_at"])
        _integer(row["sequence"])
        inbox = [
            r
            for r in builder.source[_TABLES[3]]
            if r["conversation_id"] == row["conversation_id"]
        ]
        if (
            len(inbox) > MAX_INBOX_PENDING
            or sum(len(r["body"]) for r in inbox) > MAX_INBOX_PENDING_CHARS
        ):
            raise ValueError("invalid_fleet_queue")
        values = row | {
            "conversation_id": builder.ref("conversation", row["conversation_id"]),
            "message_id": builder.ref("fleet_message", row["message_id"]),
            "sequence": builder.ref("fleet_sequence", row["sequence"]),
        }
        for name, domain in (
            ("run_id", "run"),
            ("parent_run_id", "run"),
            ("chain_id", "work_chain"),
        ):
            values[name] = builder.ref(domain, row[name], optional=name == "chain_id")
            if row[name] is not None:
                require(_TABLES[3], row, "A-runs", domain, row[name], values[name])
        builder.emit(
            _TABLES[3],
            row,
            values,
            proofs=("A-runs:fleet_lineage", "S:imported_delivery_quarantine"),
        )

    builder.handle(_TABLES[3], fleet)

    def generation(row):
        from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta

        _integer(row["position"])
        for name in ("prompt", "negative_prompt", "backend", "created_at"):
            _text(row[name])
        for name in ("model", "style"):
            _text(row[name], optional=True)
        if row["seed"] is not None:
            _integer(row["seed"], -(2**63))
        params = _object(row["params_json"])
        native = GenerationVariantMeta.from_row(row)
        if native.params != params:
            raise ValueError("lossy_generation_metadata")
        attachments = [
            r
            for r in settings.graph.rows
            if r.table == "message_attachments"
            and r.original == (row["message_id"], row["position"])
        ]
        message = originals["messages"][row["message_id"]]
        if not attachments and not (
            row["position"] == 0 and message["image_data"] is not None
        ):
            raise _UnprovedReference(
                "generation_attachment", (row["message_id"], row["position"])
            )
        builder.emit(
            _TABLES[4],
            row,
            row | {"message_id": builder.ref("message", row["message_id"])},
        )

    builder.handle(_TABLES[4], generation)

    def trajectory(row):
        from tldw_chatbook.Chat.trajectory import _sidecar_event_id

        _integer(row["seq"], -(2**63))
        _text(row["event_kind"], limit=256)
        for name in ("model", "provider"):
            _text(row[name], optional=True)
        for name in ("step_started_at", "first_token_at", "completed_at"):
            value = row[name]
            if value is not None and (
                type(value) not in (float, int) or not math.isfinite(value)
            ):
                raise ValueError("invalid_trajectory_time")
        values = row | {
            "message_id": message_ref(row),
            "conversation_id": builder.ref("conversation", row["conversation_id"]),
        }
        # The native sidecar turn groups persisted USER messages; unknown
        # ephemeral/runtime turns require T's actual source association.
        turn = originals["messages"].get(row["turn_id"])
        if turn is None:
            raise _UnprovedReference("trajectory_turn", row["turn_id"])
        if turn["conversation_id"] != row["conversation_id"]:
            raise ValueError("invalid_trajectory_turn")
        values["turn_id"] = builder.ref("message", row["turn_id"])
        values["payload_json"] = _trajectory_payload(
            builder,
            row,
            lambda handler, domain, source, mapped: require(
                _TABLES[5], row, handler, domain, source, mapped
            ),
        )
        builder.emit(_TABLES[5], row, values)
        provenance.append(
            _WorkflowProvenance(
                _TABLES[5],
                builder.original(_TABLES[5], row),
                _sidecar_event_id(row),
                _sidecar_event_id(values),
                "historical_source_event"
                if row["event_kind"] == "capture_failed"
                else "native_sidecar_event",
            )
        )

    builder.handle(_TABLES[5], trajectory)

    def annotation(row):
        _integer(row["deleted"], 0, 1)
        for name in ("quote_text", "comment", "created_at", "updated_at"):
            _text(row[name])
        _text(row["row_key"])
        if not row["row_key"].startswith("message:"):
            raise _UnprovedReference("trajectory_event", row["row_key"])
        anchor_source = row["row_key"][len("message:") :]
        anchor = "message:" + message_ref(row | {"message_id": anchor_source})
        if (
            row["message_id"] is not None
            and row["row_key"] != "message:" + row["message_id"]
        ):
            raise ValueError("annotation_anchor_mismatch")
        values = row | {
            "annotation_id": builder.ref("transcript_annotation", row["annotation_id"]),
            "conversation_id": builder.ref("conversation", row["conversation_id"]),
            "message_id": message_ref(row, optional=True),
            "row_key": anchor,
        }
        builder.emit(_TABLES[6], row, values)
        provenance.append(
            _WorkflowProvenance(
                _TABLES[6],
                builder.original(_TABLES[6], row),
                row["row_key"],
                anchor,
                "native_message_anchor",
            )
        )

    builder.handle(_TABLES[6], annotation)
    accepted_checkpoints = {
        r.original[0] for r in builder.rows if r.table == _TABLES[0]
    }
    for row in builder.source[_TABLES[0]]:
        if row["assistant_message_id"] not in accepted_checkpoints:
            builder.blockers.append(
                _ContentBlocker(
                    "messages",
                    (row["assistant_message_id"],),
                    "dependency_unproven",
                    "dispatch_checkpoint",
                    row["assistant_message_id"],
                )
            )
    checkpoint_conversations = {
        row["conversation_id"] for row in builder.source[_TABLES[0]]
    }
    conflicting_owner_paths = set()
    for conversation_id in originals["conversations"]:
        if conversation_id in checkpoint_conversations:
            continue
        active_owners = 0
        for message_id in active_path(conversation_id):
            candidate = originals["messages"][message_id]
            if candidate["role"] != "assistant" or candidate["deleted"] != 0:
                continue
            continuation = read_provider_continuation_json(
                candidate["provider_continuation_json"]
            ).checkpoint
            if continuation is not None and continuation.state == "active":
                active_owners += 1
        # Native checkpoint-free recovery rejects multiple active owners before
        # returning any individual owner. Off-path/deleted history is not counted.
        if active_owners > 1:
            conflicting_owner_paths.add(conversation_id)
            builder.blockers.append(
                _ContentBlocker(
                    "conversations", (conversation_id,), "duplicate_active_path_owner"
                )
            )
    for item in settings.successors:
        if item.handler != "C2b":
            successors.append(item)
            continue
        if item.domain in {"dispatch_owner", "provider_continuation"}:
            message = originals["messages"][item.original_id]
            if message["conversation_id"] in conflicting_owner_paths:
                continue
            private = message["provider_continuation_json"]
            continuation = (
                None if private is None else parse_provider_continuation_json(private)
            )
            state = message["assistant_generation_state"]
            if (
                message["role"] != "assistant"
                or (
                    continuation is not None
                    and continuation.state != "active"
                    and not (continuation.state == "complete" and state == "complete")
                )
                or (continuation is None and state == "continuation_active")
            ):
                builder.blockers.append(
                    _ContentBlocker(item.table, item.original, "native_content_invalid")
                )
                continue
            # Native checkpoint-free accepted/dispatched and legacy active
            # continuations are readable history; they carry no local actions.
            successors.append(replace(item, handler="S"))
        elif item.domain in {"handoff_run", "chat_start_attempt"}:
            successors.append(
                replace(
                    item,
                    handler="A-runs",
                    domain="run" if item.domain == "handoff_run" else item.domain,
                )
            )
        elif item.domain == "canvas_turn":
            successors.append(replace(item, handler="T"))
        else:
            raise ValueError("unknown_workflow_successor")
        discharged.append(item)
    blocked_rows = {(b.table, b.original) for b in builder.blockers}
    builder.rows = [
        r for r in builder.rows if (r.table, r.original) not in blocked_rows
    ]
    builder.close_dependencies()
    accepted = {(r.table, r.original) for r in builder.rows}
    closed = {(s.table, s.original, "C2b:" + s.domain) for s in discharged}
    rows = tuple(
        replace(
            r,
            required_proofs=tuple(
                p for p in r.required_proofs if (r.table, r.original, p) not in closed
            )
            + tuple(
                sorted(
                    {
                        s.handler + ":" + s.domain
                        for s in successors
                        if s.table == r.table and s.original == r.original
                    }
                )
            ),
        )
        for r in builder.rows
    )
    reads.check()
    return _WorkflowPlan(
        settings,
        rows,
        builder.retained(),
        tuple(builder.blockers),
        tuple(s for s in successors if (s.table, s.original) in accepted),
        tuple(s for s in discharged if (s.table, s.original) in accepted),
        tuple(p for p in provenance if (p.table, p.original) in accepted),
    )
