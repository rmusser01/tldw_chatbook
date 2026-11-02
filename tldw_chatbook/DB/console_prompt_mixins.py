"""Conversation-owned mix-in records; every write borrows its caller's cursor."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, replace
from uuid import uuid4

from tldw_chatbook.Chat.console_prompt_mixins import (
    MixinAttemptSnapshot,
    MixinAttemptWrite,
    MixinForkState,
    MixinPreferenceConflictError,
    MixinPreferenceRead,
    MixinPreferences,
    MixinReplayRead,
    MixinSnapshot,
    MixinSnapshotUnavailableError,
    MixinSourceRef,
    decode_mixin_preferences,
    encode_mixin_preferences,
)
from tldw_chatbook.Prompt_Management.prompt_artifact_models import (
    LiteralMixinDefinition,
)
from tldw_chatbook.Utils.input_validation import strict_json_loads

_SNAPSHOT_COLUMNS = (
    "authority_key, backend, record_uuid, record_version, name, placement, body"
)


def _require_transaction(cursor: sqlite3.Cursor) -> None:
    if not cursor.connection.in_transaction:
        raise RuntimeError("mixin_parent_transaction_required")


def read_preferences(
    cursor: sqlite3.Cursor, conversation_id: str
) -> MixinPreferenceRead:
    """Read selections without interpreting unknown formats as an empty set."""
    row = cursor.execute(
        "SELECT schema_version, revision, selection_json FROM console_prompt_mixin_preferences WHERE conversation_id=?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return decode_mixin_preferences(None)
    result = decode_mixin_preferences(row[2])
    if row[0] != 1:
        return MixinPreferenceRead("unsupported", None, row[2])
    if result.preferences is not None and result.preferences.revision != row[1]:
        return MixinPreferenceRead("corrupt", None, row[2])
    return result


def save_preferences(
    cursor: sqlite3.Cursor,
    conversation_id: str,
    *,
    expected_revision: int,
    value: MixinPreferences,
) -> MixinPreferences:
    """CAS only this preference row, preserving unknown formats and chat state."""
    _require_transaction(cursor)
    if (
        type(expected_revision) is not int
        or expected_revision < 0
        or type(value) is not MixinPreferences
    ):
        raise MixinPreferenceConflictError("mixin_preference_invalid_revision")
    current = read_preferences(cursor, conversation_id)
    if current.preferences is None or current.preferences.revision != expected_revision:
        raise MixinPreferenceConflictError("mixin_preference_conflict")
    saved = replace(value, revision=expected_revision + 1)
    raw = encode_mixin_preferences(saved)
    if current.status == "absent":
        cursor.execute(
            "INSERT INTO console_prompt_mixin_preferences(conversation_id,schema_version,revision,selection_json) VALUES(?,?,?,?)",
            (conversation_id, 1, saved.revision, raw),
        )
    else:
        result = cursor.execute(
            "UPDATE console_prompt_mixin_preferences SET revision=?,selection_json=? WHERE conversation_id=? AND revision=? AND schema_version=1",
            (saved.revision, raw, conversation_id, expected_revision),
        )
        if result.rowcount != 1:
            raise MixinPreferenceConflictError("mixin_preference_conflict")
    return saved


def _snapshot_values(item: MixinSnapshot) -> tuple[object, ...]:
    return (
        item.source.authority_key,
        item.source.backend,
        item.source.record_uuid,
        item.record_version,
        item.name,
        item.definition.placement,
        item.definition.body,
    )


def _snapshot_from_row(row) -> MixinSnapshot:
    return MixinSnapshot(
        MixinSourceRef(row[0], row[1], row[2]),
        row[3],
        row[4],
        LiteralMixinDefinition(row[6], row[5]),
    )


def _metadata(raw: str | None) -> dict:
    if raw is None:
        return {}
    try:
        value = strict_json_loads(raw)
        if type(value) is dict:
            return value
    except (ValueError, TypeError):
        pass
    raise MixinSnapshotUnavailableError("mixin_invalid_anchor_metadata")


def _mark_assistant(
    cursor, conversation_id, assistant_id, attempt_id, predecessor=None
) -> None:
    row = cursor.execute(
        "SELECT metadata_json FROM messages WHERE conversation_id=? AND id=? AND sender='assistant'",
        (conversation_id, assistant_id),
    ).fetchone()
    if row is None:
        raise MixinSnapshotUnavailableError("mixin_anchor_unavailable")
    metadata = _metadata(row[0])
    if (
        metadata.get("mixin_attempt_id", attempt_id) not in {attempt_id, predecessor}
        or metadata.get("mixin_snapshot_schema_version", 1) != 1
    ):
        raise MixinSnapshotUnavailableError("mixin_anchor_attempt_conflict")
    metadata.update(mixin_attempt_id=attempt_id, mixin_snapshot_schema_version=1)
    cursor.execute(
        "UPDATE messages SET metadata_json=? WHERE conversation_id=? AND id=?",
        (json.dumps(metadata, sort_keys=True), conversation_id, assistant_id),
    )


def write_attempt(
    cursor: sqlite3.Cursor,
    conversation_id: str,
    *,
    batch: MixinAttemptWrite,
    native_message_ids: Mapping[str, str],
) -> None:
    """Retain exact instructions and body-free markers in the parent transaction."""
    _require_transaction(cursor)
    if type(batch) is not MixinAttemptWrite:
        raise MixinSnapshotUnavailableError("mixin_invalid_attempt")
    try:
        user_id = native_message_ids[batch.user_message_id]
        assistant_id = native_message_ids[batch.assistant_message_id]
    except KeyError:
        raise MixinSnapshotUnavailableError("mixin_native_anchor_missing") from None
    for message_id, sender in ((user_id, "user"), (assistant_id, "assistant")):
        if (
            cursor.execute(
                "SELECT 1 FROM messages WHERE id=? AND conversation_id=? AND sender=?",
                (message_id, conversation_id, sender),
            ).fetchone()
            is None
        ):
            raise MixinSnapshotUnavailableError("mixin_anchor_ownership")
    existing = cursor.execute(
        "SELECT user_message_id,assistant_message_id,retry_of_attempt_id,replaces_attempt_id FROM console_prompt_mixin_attempts WHERE conversation_id=? AND attempt_id=?",
        (conversation_id, batch.attempt_id),
    ).fetchone()
    if existing is not None:
        replay = read_attempt(cursor, conversation_id, attempt_id=batch.attempt_id)
        if (
            tuple(existing)
            != (
                user_id,
                assistant_id,
                batch.retry_of_attempt_id,
                batch.replaces_attempt_id,
            )
            or replay.status != "recorded"
            or replay.snapshot != batch.snapshot
        ):
            raise MixinSnapshotUnavailableError("mixin_attempt_conflict")
        return
    if (
        batch.retry_of_attempt_id is not None
        and read_attempt(
            cursor, conversation_id, attempt_id=batch.retry_of_attempt_id
        ).status
        != "recorded"
    ):
        raise MixinSnapshotUnavailableError("mixin_retry_predecessor_unavailable")
    if batch.replaces_attempt_id is not None:
        predecessor = cursor.execute(
            "SELECT user_message_id FROM console_prompt_mixin_attempts WHERE conversation_id=? AND attempt_id=?",
            (conversation_id, batch.replaces_attempt_id),
        ).fetchone()
        if (
            predecessor is None
            or predecessor[0] != user_id
            or read_attempt(
                cursor, conversation_id, attempt_id=batch.replaces_attempt_id
            ).status
            != "recorded"
        ):
            raise MixinSnapshotUnavailableError(
                "mixin_replacement_predecessor_unavailable"
            )
    # Validate metadata before writing any rows; callers still own rollback.
    metadata = _metadata(
        cursor.execute(
            "SELECT metadata_json FROM messages WHERE id=?", (assistant_id,)
        ).fetchone()[0]
    )
    if (
        "mixin_attempt_id" in metadata or "mixin_snapshot_schema_version" in metadata
    ) and (
        batch.retry_of_attempt_id is None
        or metadata.get("mixin_attempt_id") != batch.retry_of_attempt_id
        or type(metadata.get("mixin_snapshot_schema_version")) is not int
        or metadata["mixin_snapshot_schema_version"] != 1
    ):
        raise MixinSnapshotUnavailableError("mixin_anchor_attempt_conflict")
    checkpoint = cursor.execute(
        "SELECT attempt_id,user_message_id FROM console_dispatch_checkpoints WHERE conversation_id=? AND assistant_message_id=?",
        (conversation_id, assistant_id),
    ).fetchone()
    if checkpoint is not None and tuple(checkpoint) != (batch.attempt_id, user_id):
        raise MixinSnapshotUnavailableError("mixin_checkpoint_ownership")
    cursor.execute(
        "INSERT INTO console_prompt_mixin_attempts(conversation_id,attempt_id,schema_version,user_message_id,assistant_message_id,retry_of_attempt_id,expected_item_count,replaces_attempt_id) VALUES(?,?,?,?,?,?,?,?)",
        (
            conversation_id,
            batch.attempt_id,
            1,
            user_id,
            assistant_id,
            batch.retry_of_attempt_id,
            len(batch.snapshot.items),
            batch.replaces_attempt_id,
        ),
    )
    _write_snapshot_items(cursor, conversation_id, batch.attempt_id, batch.snapshot)
    if checkpoint is not None:
        cursor.execute(
            "UPDATE console_dispatch_checkpoints SET mixin_snapshot_schema_version=1 WHERE conversation_id=? AND assistant_message_id=?",
            (conversation_id, assistant_id),
        )
    _mark_assistant(
        cursor,
        conversation_id,
        assistant_id,
        batch.attempt_id,
        batch.retry_of_attempt_id,
    )


def _write_snapshot_items(cursor, conversation_id, attempt_id, snapshot):
    for ordinal, item in enumerate(snapshot.items):
        values = _snapshot_values(item)
        digest = hashlib.sha256(
            json.dumps(asdict(item), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        candidates = cursor.execute(
            "SELECT snapshot_id,"
            + _SNAPSHOT_COLUMNS
            + " FROM console_prompt_mixin_revisions WHERE conversation_id=? AND digest=?",
            (conversation_id, digest),
        ).fetchall()
        snapshot_id = next(
            (row[0] for row in candidates if tuple(row)[1:] == values), None
        )
        if snapshot_id is None:
            snapshot_id = str(uuid4())
            cursor.execute(
                "INSERT INTO console_prompt_mixin_revisions(conversation_id,snapshot_id,digest,"
                + _SNAPSHOT_COLUMNS
                + ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                (conversation_id, snapshot_id, digest, *values),
            )
        cursor.execute(
            "INSERT INTO console_prompt_mixin_attempt_items(conversation_id,attempt_id,ordinal,snapshot_id) VALUES(?,?,?,?)",
            (conversation_id, attempt_id, ordinal, snapshot_id),
        )


def read_attempt(
    cursor: sqlite3.Cursor, conversation_id: str, *, attempt_id: str
) -> MixinReplayRead:
    """Distinguish legacy absence from recorded-empty and damaged new records."""
    row = cursor.execute(
        "SELECT schema_version,expected_item_count FROM console_prompt_mixin_attempts WHERE conversation_id=? AND attempt_id=?",
        (conversation_id, attempt_id),
    ).fetchone()
    if row is None:
        marked = cursor.execute(
            "SELECT 1 FROM messages WHERE conversation_id=? AND CASE WHEN json_valid(metadata_json) THEN json_extract(metadata_json,'$.mixin_attempt_id')=? ELSE 0 END LIMIT 1",
            (conversation_id, attempt_id),
        ).fetchone()
        checkpoint = cursor.execute(
            "SELECT 1 FROM console_dispatch_checkpoints WHERE conversation_id=? AND attempt_id=? AND mixin_snapshot_schema_version IS NOT NULL",
            (conversation_id, attempt_id),
        ).fetchone()
        return MixinReplayRead(
            "unavailable" if marked or checkpoint else "legacy", None
        )
    if row[0] != 1:
        return MixinReplayRead("unavailable", None)
    rows = cursor.execute(
        "SELECT i.ordinal,"
        + ",".join("r." + name.strip() for name in _SNAPSHOT_COLUMNS.split(","))
        + " FROM console_prompt_mixin_attempt_items i LEFT JOIN console_prompt_mixin_revisions r ON r.conversation_id=i.conversation_id AND r.snapshot_id=i.snapshot_id WHERE i.conversation_id=? AND i.attempt_id=? ORDER BY i.ordinal",
        (conversation_id, attempt_id),
    ).fetchall()
    if len(rows) != row[1] or any(
        item[0] != ordinal for ordinal, item in enumerate(rows)
    ):
        return MixinReplayRead("unavailable", None)
    try:
        snapshot = MixinAttemptSnapshot(
            tuple(_snapshot_from_row(tuple(item)[1:]) for item in rows)
        )
    except (ValueError, TypeError):
        return MixinReplayRead("unavailable", None)
    return MixinReplayRead("recorded", snapshot)


def read_fork_state(
    cursor: sqlite3.Cursor, conversation_id: str, message_ids: Mapping[str, str]
) -> MixinForkState:
    """Read a coherent historical prefix using the same caller-owned transaction."""
    from tldw_chatbook.Chat.console_prompt_mixins import MixinForkAttempt

    rows = cursor.execute(
        "SELECT attempt_id,user_message_id,assistant_message_id,retry_of_attempt_id FROM console_prompt_mixin_attempts WHERE conversation_id=?",
        (conversation_id,),
    ).fetchall()
    by_id = {row[0]: row for row in rows}
    selected = {row[0] for row in rows if row[2] in message_ids}
    pending = list(selected)
    while pending:
        predecessor = by_id[pending.pop()][3]
        if predecessor is not None and predecessor not in selected:
            if predecessor not in by_id:
                raise MixinSnapshotUnavailableError(
                    "mixin_retry_predecessor_unavailable"
                )
            selected.add(predecessor)
            pending.append(predecessor)
    attempts = []
    for row in rows:
        if row[0] not in selected:
            continue
        replay = read_attempt(cursor, conversation_id, attempt_id=row[0])
        if replay.status != "recorded" or replay.snapshot is None:
            raise MixinSnapshotUnavailableError("mixin_copy_unavailable")
        attempts.append(MixinForkAttempt(*tuple(row), replay.snapshot))
    markers = []
    for message_id in message_ids:
        row = cursor.execute(
            "SELECT metadata_json,sender FROM messages WHERE conversation_id=? AND id=?",
            (conversation_id, message_id),
        ).fetchone()
        if row is None:
            raise MixinSnapshotUnavailableError("mixin_anchor_unavailable")
        if row[1] != "assistant":
            continue
        try:
            metadata = _metadata(row[0])
        except MixinSnapshotUnavailableError:
            # Unreadable metadata proves neither a receipt nor legacy absence.
            metadata = {"mixin_snapshot_schema_version": None}
        markers.append(
            (
                message_id,
                json.dumps(
                    {
                        key: metadata[key]
                        for key in ("mixin_attempt_id", "mixin_snapshot_schema_version")
                        if key in metadata
                    },
                    sort_keys=True,
                ),
            )
        )
    return MixinForkState(tuple(attempts), tuple(markers)).remap(message_ids)


def write_fork_state(
    cursor: sqlite3.Cursor,
    conversation_id: str,
    *,
    state: MixinForkState,
    message_ids: Mapping[str, str],
) -> None:
    """Import owned history once, then publish each exact source marker once."""
    _require_transaction(cursor)
    if type(state) is not MixinForkState:
        raise MixinSnapshotUnavailableError("mixin_invalid_fork_state")
    for row in state.attempts:
        for anchor, sender in (
            (row.user_message_id, "user"),
            (row.assistant_message_id, "assistant"),
        ):
            if anchor is not None and (
                anchor not in message_ids
                or cursor.execute(
                    "SELECT 1 FROM messages WHERE conversation_id=? AND id=? AND sender=?",
                    (conversation_id, message_ids[anchor], sender),
                ).fetchone()
                is None
            ):
                raise MixinSnapshotUnavailableError("mixin_anchor_ownership")
    for row in state.attempts:
        cursor.execute(
            "INSERT INTO console_prompt_mixin_attempts(conversation_id,attempt_id,schema_version,user_message_id,assistant_message_id,retry_of_attempt_id,expected_item_count) VALUES(?,?,?,?,?,?,?)",
            (
                conversation_id,
                row.attempt_id,
                1,
                message_ids.get(row.user_message_id),
                message_ids.get(row.assistant_message_id),
                row.retry_of_attempt_id,
                len(row.snapshot.items),
            ),
        )
        _write_snapshot_items(cursor, conversation_id, row.attempt_id, row.snapshot)
    for message_id, raw in state.markers:
        destination = message_ids.get(message_id)
        row = cursor.execute(
            "SELECT metadata_json FROM messages WHERE conversation_id=? AND id=? AND sender='assistant'",
            (conversation_id, destination),
        ).fetchone()
        if row is None:
            raise MixinSnapshotUnavailableError("mixin_anchor_ownership")
        metadata = _metadata(row[0])
        marker = strict_json_loads(raw)
        existing = {
            key: metadata[key]
            for key in ("mixin_attempt_id", "mixin_snapshot_schema_version")
            if key in metadata
        }
        if existing and existing != marker:
            raise MixinSnapshotUnavailableError("mixin_anchor_attempt_conflict")
        metadata.update(marker)
        cursor.execute(
            "UPDATE messages SET metadata_json=? WHERE conversation_id=? AND id=?",
            (json.dumps(metadata, sort_keys=True), conversation_id, destination),
        )


def copy_conversation_mixins(
    cursor: sqlite3.Cursor,
    *,
    source_conversation_id: str,
    destination_conversation_id: str,
    message_ids: Mapping[str, str],
) -> None:
    """Copy used records into the fork's ownership without active preferences."""
    _require_transaction(cursor)
    if source_conversation_id == destination_conversation_id:
        raise MixinSnapshotUnavailableError("mixin_copy_same_owner")
    state = read_fork_state(
        cursor, source_conversation_id, {key: key for key in message_ids}
    )
    write_fork_state(
        cursor, destination_conversation_id, state=state, message_ids=message_ids
    )


def bind_attempt_trace_call(
    cursor: sqlite3.Cursor, *, conversation_id: str, user_message_id: str, call_id: str
) -> None:
    """Bind the first captured call using its still-owned native dispatch checkpoint."""
    _require_transaction(cursor)
    checkpoint = cursor.execute(
        "SELECT attempt_id,assistant_message_id FROM console_dispatch_checkpoints "
        "WHERE conversation_id=? AND user_message_id=? AND mixin_snapshot_schema_version=1",
        (conversation_id, user_message_id),
    ).fetchall()
    if not checkpoint:
        return
    if len(checkpoint) != 1:
        raise MixinSnapshotUnavailableError("mixin_trace_owner_unavailable")
    attempt_id, assistant_id = checkpoint[0]
    changed = cursor.execute(
        "UPDATE console_prompt_mixin_attempts SET trace_call_id=? "
        "WHERE conversation_id=? AND attempt_id=? AND assistant_message_id=? "
        "AND (trace_call_id IS NULL OR trace_call_id=?)",
        (call_id, conversation_id, attempt_id, assistant_id, call_id),
    ).rowcount
    if changed != 1:
        raise MixinSnapshotUnavailableError("mixin_trace_owner_unavailable")


def seal_attempt_result(
    cursor: sqlite3.Cursor,
    *,
    conversation_id: str,
    assistant_message_id: str,
    metadata_json: str | None,
) -> None:
    """Seal the exact complete replacement revision inside terminal settlement."""
    _require_transaction(cursor)
    metadata = _metadata(metadata_json)
    attempt_id = metadata.get("mixin_attempt_id")
    if attempt_id is None:
        return
    attempt = cursor.execute(
        "SELECT replaces_attempt_id,result_revision_id FROM console_prompt_mixin_attempts "
        "WHERE conversation_id=? AND attempt_id=? AND assistant_message_id=?",
        (conversation_id, attempt_id, assistant_message_id),
    ).fetchone()
    if attempt is None:
        raise MixinSnapshotUnavailableError("mixin_result_owner_unavailable")
    if attempt[0] is None:
        return
    revision = cursor.execute(
        "SELECT revision_id FROM console_trace_semantic_revisions WHERE live_message_id=? "
        "ORDER BY revision_sequence DESC LIMIT 1",
        (assistant_message_id,),
    ).fetchone()
    if revision is None or (attempt[1] is not None and attempt[1] != revision[0]):
        raise MixinSnapshotUnavailableError("mixin_result_owner_unavailable")
    cursor.execute(
        "UPDATE console_prompt_mixin_attempts SET result_revision_id=? WHERE conversation_id=? AND attempt_id=?",
        (revision[0], conversation_id, attempt_id),
    )
