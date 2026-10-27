"""Fixed native Sync frontiers; proof transformations confer no write authority."""

import hashlib
import json
from dataclasses import dataclass, field
from itertools import chain

from tldw_chatbook.DB.ChaChaNotes_DB import (
    _normalize_legacy_chat_delete_intent_payload,
    _normalize_legacy_chat_sync_intent_payload,
    _validated_thinking_blocks_json,
)
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash

from .mobile_content_graph import _ContentBlocker, _UnprovedReference
from .mobile_identity_plan import _retired_world_entry
from .mobile_workspace_plan import _object

_DOMAINS = {
    "messages": "message",
    "conversations": "conversation",
    "notes": "note",
    "character_cards": "character",
    "keywords": "keyword",
    "keyword_collections": "keyword_collection",
    "chat_dictionaries": "dictionary",
    "world_books": "world_book",
    "world_book_entries": "world_book_entry",
}
# Installed trigger payloads, not arbitrary JSON or a recursive reference grammar.
_PAYLOAD_COLUMNS = {
    "messages": "id conversation_id parent_message_id sender content image_mime_type provider_continuation_json thinking_blocks_json assistant_generation_state timestamp ranking last_modified deleted client_id version",
    "conversations": "id root_id forked_from_message_id parent_conversation_id character_id assistant_kind assistant_id persona_memory_mode scope_type workspace_id state topic_label topic_label_source topic_last_tagged_at topic_last_tagged_message_id cluster_id source external_ref runtime_backend discovery_owner discovery_entity_id system_prompt metadata thinking_history_policy title rating created_at last_modified deleted client_id version",
    "notes": "id title content created_at last_modified deleted client_id version",
    "character_cards": "id name description personality scenario post_history_instructions first_message message_example creator_notes system_prompt alternate_greetings tags creator character_version extensions created_at last_modified deleted client_id version",
    "keywords": "id keyword created_at last_modified deleted client_id version",
    "keyword_collections": "id name parent_id created_at last_modified deleted client_id version",
    "chat_dictionaries": "id name description file_path strategy max_tokens enabled created_at last_modified deleted client_id version",
    "world_books": "id name description scan_depth token_budget recursive_scanning enabled created_at last_modified deleted client_id version",
    "world_book_entries": "id world_book_id keys content enabled position insertion_order priority selective secondary_keys case_sensitive regex extensions created_at last_modified",
}
_DELETE_COLUMNS = ("id", "deleted", "last_modified", "version", "client_id")
_SYNC_COLUMNS = (
    "change_id",
    "entity",
    "entity_id",
    "operation",
    "timestamp",
    "client_id",
    "version",
    "payload",
)


class _SyncRefusal(ValueError):
    """A fixed body-free frontier diagnosis."""


@dataclass(frozen=True)
class _SyncProvenance:
    change_id: int
    entity: str
    original_id: str
    operation: str
    timestamp: str
    client_id: str
    version: int
    source_digest: str
    source_manifest_digest: str


@dataclass(frozen=True)
class _SyncFrontier:
    entity: str
    original_id: str | int
    mapped_id: str | int
    version: int
    operation: str
    timestamp: str
    client_id: str
    payload: str = field(repr=False)
    payload_hash: str | None
    previous_payload: str | None = field(repr=False)
    previous_operation: str | None
    previous_timestamp: str | None
    previous_client_id: str | None
    previous_hash: str | None
    # Native live INSERT owns current proof. The writer must replace its deleted
    # create proof atomically; this plan cannot authorize that lifecycle.
    current_disposition: str
    source_change_ids: tuple[int, ...]
    owner: tuple[str, tuple]


@dataclass(frozen=True)
class _SyncPlan:
    trace: object = field(repr=False)
    frontiers: tuple[_SyncFrontier, ...] = field(repr=False)
    provenance: tuple[_SyncProvenance, ...]
    blockers: tuple[_ContentBlocker, ...]
    field_ownership: tuple = (("sync_log", _SYNC_COLUMNS),)
    pending_handlers: tuple[str, ...] = ("integrated", "owner-write", "A-runs")

    @property
    def identities(self):
        return self.trace.identities

    def require_complete(self):
        raise _SyncRefusal("mobile_sync_handlers_pending")


def _json(value, budget):
    budget.value(value, text_copies=2)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _message_state(payload, role, budget, *, captured_delete=False):
    """Preserve the native continuation/state predicate, including tombstones."""
    from tldw_chatbook.Chat.assistant_generation_state import (
        normalize_assistant_generation_state,
    )
    from tldw_chatbook.Chat.provider_continuation import (
        dump_provider_continuation_json,
        parse_provider_continuation_json,
    )

    if type(role) is not str:
        raise _SyncRefusal("message_payload_unrepresentable")
    private = payload["provider_continuation_json"]
    active = False
    if private is not None:
        if role != "assistant" or type(private) is not str:
            raise _SyncRefusal("continuation_unrepresentable")
        budget.json(private)
        budget.value(private, text_copies=2)
        checkpoint = parse_provider_continuation_json(private)
        if dump_provider_continuation_json(checkpoint) != private:
            raise _SyncRefusal("continuation_unrepresentable")
        active = checkpoint.state == "active"
    if payload["assistant_generation_state"] is not None and role != "assistant":
        raise _SyncRefusal("generation_state_unrepresentable")
    state = normalize_assistant_generation_state(
        role=role,
        raw_state=payload["assistant_generation_state"],
        has_valid_active_continuation=active,
    )
    if (
        state is not None
        and state.value == "continuation_active"
        and not active
        and not captured_delete
    ):
        raise _SyncRefusal("continuation_unrepresentable")
    return state.value if state else None, private


def _message_hash(payload, role, budget):
    """Use the native envelope, continuation and thinking canonicalizers."""
    if type(payload.get("content")) is not str:
        raise _SyncRefusal("message_payload_unrepresentable")
    state, private = _message_state(payload, role, budget)
    budget.container(5)
    envelope = {
        "assistant_generation_state": state,
        "content": payload["content"],
        "role": role,
    }
    if private is not None:
        envelope["provider_continuation_json"] = private
    thinking = payload["thinking_blocks_json"]
    if thinking is not None:
        budget.json(thinking)
        budget.value(thinking, text_copies=2)
        if role != "assistant" or _validated_thinking_blocks_json(thinking) != thinking:
            raise _SyncRefusal("thinking_unrepresentable")
        envelope["thinking_blocks_json"] = thinking
    budget.value(envelope, text_copies=2)
    return canonical_payload_hash(envelope)


def _tombstone(payload, entity, original, mapped, budget):
    budget.container(len(payload))
    if entity == "messages":
        value = _normalize_legacy_chat_delete_intent_payload(payload)
        if value is None:
            raise _SyncRefusal("delete_payload_unrepresentable")
        expected = {key: original[key] for key in _DELETE_COLUMNS}
        expected["assistant_generation_state"], _private = _message_state(
            original,
            original["role"],
            budget,
            captured_delete=value["base_payload_hash"] is not None,
        )
        if any(value[key] != expected[key] for key in expected):
            raise _SyncRefusal("delete_payload_mismatch")
        if value["base_payload_hash"] is None:
            if not value["legacy_pre_v50_base_reconstruction"]:
                raise _SyncRefusal("delete_base_proof_missing")
            budget.container(len(original))
            # Native pre-v50 reconstruction deliberately predates thinking hashes.
            value["base_payload_hash"] = _message_hash(
                original | {"thinking_blocks_json": None}, original["role"], budget
            )
            value["legacy_pre_v50_base_reconstruction"] = False
        value["id"] = mapped["id"]
        return value
    if set(payload) != set(_DELETE_COLUMNS) or payload != {
        key: original[key] for key in _DELETE_COLUMNS
    }:
        raise _SyncRefusal("delete_payload_mismatch")
    return payload | {"id": mapped["id"]}


def _previous(entry, original, identities, budget):
    budget.json(entry["payload"])
    payload = _object(entry["payload"])
    budget.container(len(payload))
    if entry["operation"] == "delete":
        payload = _normalize_legacy_chat_delete_intent_payload(payload)
        if (
            payload is None
            or payload["id"] != original["id"]
            or payload["version"] != original["version"] - 1
            or payload["deleted"] != 1
        ):
            raise _SyncRefusal("previous_delete_proof_invalid")
        budget.container(2)
        _message_state(
            {
                "provider_continuation_json": original["provider_continuation_json"],
                "assistant_generation_state": payload["assistant_generation_state"],
            },
            original["role"],
            budget,
        )
        payload["id"] = identities.mapped("message", original["id"])
        return payload, canonical_payload_hash({"deleted": True})
    payload = _normalize_legacy_chat_sync_intent_payload(payload)
    if (
        entry["operation"] not in {"create", "update"}
        or payload is None
        or payload["id"] != original["id"]
        or payload["conversation_id"] != original["conversation_id"]
        or payload["version"] != original["version"] - 1
        or payload["deleted"] != 0
    ):
        raise _SyncRefusal("previous_proof_invalid")
    digest = _message_hash(payload, original["role"], budget)
    for column, domain in (
        ("id", "message"),
        ("conversation_id", "conversation"),
        ("parent_message_id", "message"),
    ):
        if payload[column] is not None:
            try:
                payload[column] = identities.mapped(domain, payload[column])
            except ValueError:
                raise _UnprovedReference(domain, payload[column]) from None
    return payload, digest


def _sync_rows(reads, budget):
    result, offset = [], 0
    while True:
        columns, rows = reads.page(
            "legacy", "chat", "sync_log", offset=offset, limit=1000
        )
        for row in rows:
            budget.value(row)
            budget.container(len(columns))
            result.append(dict(zip(columns, row, strict=True)))
        if len(rows) < 1000:
            budget.container(len(result))
            return tuple(result)
        offset += len(rows)


def _plan_sync(reads, *, _read_budget=None):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise _SyncRefusal("mobile_sync_native_snapshot_required")
    reads.check()
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .mobile_trace_plan import _plan_traces

    budget = _TraceReadBudget() if _read_budget is None else _read_budget
    trace = _plan_traces(reads, _read_budget=budget)
    for row in chain(
        trace.content.source_rows, trace.content.rows, trace.authority_discharges
    ):
        if row.table in _DOMAINS:
            budget.container(len(row.values))
            budget.container()
    originals = {
        (r.table, r.original[0]): dict(r.values)
        for r in trace.content.source_rows
        if r.table in _DOMAINS
    }
    mapped = {
        (r.table, r.original[0]): dict(r.values)
        for r in trace.content.rows
        if r.table in _DOMAINS
    }
    mapped.update(
        {(r.table, r.original[0]): dict(r.values) for r in trace.authority_discharges}
    )
    entries = _sync_rows(reads, budget)
    groups, provenance, blockers, frontiers = {}, [], [], []
    for row in entries:
        if tuple(row) != _SYNC_COLUMNS:
            raise _SyncRefusal("mobile_sync_schema_mismatch")
        budget.container(len(row))
        budget.container()
        provenance.append(
            _SyncProvenance(
                *(row[key] for key in _SYNC_COLUMNS[:-1]),
                hashlib.sha256(_json(tuple(row.items()), budget).encode()).hexdigest(),
                trace.identities.source_manifest_digest,
            )
        )
        if (
            row["entity"] not in _DOMAINS
            or row["operation"] not in {"create", "update", "delete"}
            or type(row["entity_id"]) is not str
            or type(row["version"]) is not int
            or row["version"] < 1
        ):
            blockers.append(
                _ContentBlocker("sync_log", (row["change_id"],), "sync_record_invalid")
            )
            continue
        budget.container()
        groups.setdefault((row["entity"], row["entity_id"]), []).append(row)
    # Native unversioned hard-delete proofs survive even alongside a new live row.
    budget.container(len(groups))
    for key, group in tuple(groups.items()):
        if key[0] != "world_book_entries":
            continue
        for row in group:
            if row["operation"] != "delete":
                continue
            budget.json(row["payload"])
            payload = _retired_world_entry(row)
            identity, parent = payload["id"], payload["world_book_id"]
            if ("world_books", parent) not in mapped:
                blockers.append(
                    _ContentBlocker(
                        "sync_log",
                        (row["change_id"],),
                        "sync_dependency_unproven",
                        "world_book",
                        parent,
                    )
                )
                continue
            target = trace.identities.mapped("world_book_entry", identity)
            budget.container(20)
            frontiers.append(
                _SyncFrontier(
                    "world_book_entries",
                    identity,
                    target,
                    1,
                    "delete",
                    row["timestamp"],
                    row["client_id"],
                    _json(
                        {
                            "id": target,
                            "world_book_id": mapped["world_books", parent]["id"],
                        },
                        budget,
                    ),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "retain_native_body_free_hard_delete",
                    (row["change_id"],),
                    ("world_books", (parent,)),
                )
            )
        budget.container(len(group))
        groups[key] = [row for row in group if row["operation"] != "delete"]
        if not groups[key]:
            del groups[key]
    for (entity, identity), original in originals.items():
        group = groups.pop((entity, str(identity)), [])
        try:
            target = mapped.get((entity, identity))
            if target is None:
                raise _UnprovedReference(_DOMAINS[entity], identity)
            if not group:
                raise _SyncRefusal("current_proof_missing")
            version = original.get("version", 1)
            deleted = original.get("deleted", 0) == 1
            latest = entity in {
                "chat_dictionaries",
                "world_books",
                "world_book_entries",
            }
            current_version = (
                max(r["version"] for r in group)
                if entity == "conversations"
                else version
            )
            current = (
                [max(group, key=lambda r: r["change_id"])]
                if latest
                else [r for r in group if r["version"] == current_version]
            )
            if len(current) != 1 or current_version > version:
                raise _SyncRefusal("current_proof_ambiguous")
            current = current[0]
            if latest and current["version"] != version:
                raise _SyncRefusal("current_proof_version_mismatch")
            budget.container(len(group))
            previous = [
                r
                for r in group
                if entity == "messages"
                and not deleted
                and version > 1
                and r["version"] == version - 1
            ]
            if (
                entity == "messages"
                and not deleted
                and version > 1
                and len(previous) != 1
            ):
                raise _SyncRefusal("previous_proof_missing_or_ambiguous")
            if len(group) != 1 + len(previous):
                raise _SyncRefusal("frontier_retention_unrepresentable")
            budget.json(current["payload"])
            payload = _object(current["payload"])
            budget.container(len(payload))
            if deleted:
                if current["operation"] != "delete":
                    raise _SyncRefusal("deleted_body_proof_forbidden")
                payload = _tombstone(payload, entity, original, target, budget)
                digest = (
                    canonical_payload_hash({"deleted": True})
                    if entity == "messages"
                    else None
                )
                disposition = "replace_native_create_with_body_free_tombstone"
            else:
                if current["operation"] not in {"create", "update"}:
                    raise _SyncRefusal("live_delete_proof_invalid")
                if entity == "messages":
                    payload = _normalize_legacy_chat_sync_intent_payload(payload)
                    if payload is None:
                        raise _SyncRefusal("current_payload_invalid")
                budget.container(len(_PAYLOAD_COLUMNS[entity]))
                columns = _PAYLOAD_COLUMNS[entity].split()
                budget.container(len(columns))
                budget.container(len(columns))
                expected = {key: original[key] for key in columns}
                # Archive-only local bumps retain the last emitted shared version.
                if entity == "conversations" and current_version < version:
                    expected["version"] = current_version
                    expected["last_modified"] = current["timestamp"]
                if payload != expected:
                    raise _SyncRefusal("current_payload_mismatch")
                payload = {key: target[key] for key in columns}
                digest = (
                    _message_hash(payload, target["role"], budget)
                    if entity == "messages"
                    else None
                )
                disposition = "native_insert_current_unique"
            prior_payload = prior_hash = None
            if previous:
                prior_payload, prior_hash = _previous(
                    previous[0], original, trace.identities, budget
                )
            budget.container(20)
            frontiers.append(
                _SyncFrontier(
                    entity,
                    identity,
                    target["id"],
                    version,
                    "delete" if deleted else "create",
                    target["last_modified"],
                    current["client_id"]
                    if entity == "world_book_entries"
                    else target["client_id"],
                    _json(payload, budget),
                    digest,
                    _json(prior_payload, budget) if prior_payload else None,
                    previous[0]["operation"] if previous else None,
                    previous[0]["timestamp"] if previous else None,
                    previous[0]["client_id"] if previous else None,
                    prior_hash,
                    disposition,
                    tuple(row["change_id"] for row in group),
                    (entity, (identity,)),
                )
            )
        except _UnprovedReference as error:
            blockers.append(
                _ContentBlocker(
                    entity,
                    (identity,),
                    "sync_dependency_unproven",
                    error.domain,
                    error.identity,
                )
            )
        except _SyncRefusal as error:
            blockers.append(_ContentBlocker(entity, (identity,), str(error)))
        except (TypeError, ValueError, KeyError):
            blockers.append(
                _ContentBlocker(entity, (identity,), "sync_frontier_unrepresentable")
            )
    for group in groups.values():
        for row in group:
            blockers.append(
                _ContentBlocker(
                    "sync_log",
                    (row["change_id"],),
                    "sync_owner_unproven",
                    row["entity"],
                    row["entity_id"],
                )
            )
    from .mobile_trace_inputs import _check_trace_members

    _check_trace_members(reads)
    budget.container(len(frontiers))
    budget.container(len(provenance))
    budget.container(len(blockers))
    return _SyncPlan(trace, tuple(frontiers), tuple(provenance), tuple(blockers))
