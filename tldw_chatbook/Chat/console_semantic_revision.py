"""Transaction-scoped semantic revision coordination for canonical messages."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
import json
import re
import sqlite3
from typing import Protocol, cast

from tldw_chatbook.Chat.console_trace_models import new_opaque_id
from tldw_chatbook.Chat.console_trace_redaction import (
    CREDENTIAL_FILTER_VERSION,
    CredentialSanitizer,
    CredentialSanitizationResult,
)
from tldw_chatbook.Chat.console_trace_repository import (
    ConsoleTraceRepository,
    SemanticRevisionRecord,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


_TOKEN_CHARS = re.compile(r"[^a-z0-9]+")
_MUTATION_OPERATIONS = frozenset(
    {
        "locator_retire",
        "message_update",
        "message_delete",
        "attachment_insert",
        "attachment_update",
        "attachment_delete",
    }
)
_INTERNAL_ENVELOPE_KEYS = frozenset(
    {"message_id", "conversation_id", "parent_message_id"}
)
_TRACE_REDACTION_POLICY_UNSUPPORTED = "trace_redaction_policy_unsupported"


def project_semantic_revision_provider_message(
    cursor: sqlite3.Cursor,
    *,
    revision_id: str,
    expected_conversation_id: str,
    policy_id: str | None = None,
) -> dict[str, object]:
    """Project a live or policy-materialized revision to provider-message shape.

    The ownership check precedes the content read. The complete canonical
    semantic envelope is decoded before projection so malformed sidecars or a
    mismatched live locator cannot be mistaken for the referenced revision.
    """

    repository = ConsoleTraceRepository()
    revision = repository.get_semantic_revision(cursor, revision_id)
    if revision is None or revision.source_conversation_id != expected_conversation_id:
        raise ValueError("revision_owner_mismatch")
    if revision.live_message_id is None:
        if policy_id is None:
            raise ValueError("revision_owner_mismatch")
        binding = repository.get_revision_policy_binding(
            cursor,
            revision_id=revision_id,
            policy_id=policy_id,
        )
        if binding is None or binding.artifact_id is None:
            raise ValueError("semantic_revision_materialization_unavailable")
        artifact = repository.get_artifact(cursor, binding.artifact_id)
        if artifact is None:
            raise ValueError("semantic_revision_materialization_unavailable")
        try:
            envelope = json.loads(artifact.sanitized_bytes)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ValueError("semantic_revision_materialization_unavailable") from exc
        if not isinstance(envelope, dict):
            raise ValueError("semantic_revision_materialization_unavailable")
        return _project_revision_envelope(revision, envelope)
    envelope = SemanticRevisionCoordinator._message_envelope(
        cursor, revision.live_message_id
    )
    if (
        envelope["message_id"] != revision.source_message_id
        or envelope["conversation_id"] != expected_conversation_id
    ):
        raise ValueError("semantic_revision_locator_mismatch")
    return _project_revision_envelope(revision, envelope)


def project_semantic_revision_provider_messages(
    cursor: sqlite3.Cursor,
    *,
    revision_ids: tuple[str, ...],
    expected_conversation_id: str,
) -> dict[str, dict[str, object]]:
    """Batch-project a bounded revision set without per-reference SQL reads."""

    if not revision_ids:
        return {}
    if len(revision_ids) > 256 or len(set(revision_ids)) != len(revision_ids):
        raise ValueError("semantic_revision_batch")
    placeholders = ",".join("?" for _ in revision_ids)
    revision_rows = cursor.execute(
        f"""SELECT revision_id, source_conversation_id, source_message_id,
                   revision_sequence, normalized_role, content_kind,
                   creation_reason, predecessor_revision_id, live_message_id,
                   live_locator_retired_at
              FROM console_trace_semantic_revisions
             WHERE revision_id IN ({placeholders})""",
        revision_ids,
    ).fetchall()
    revisions = {str(row[0]): SemanticRevisionRecord(*row) for row in revision_rows}
    if len(revisions) != len(revision_ids) or any(
        revision.source_conversation_id != expected_conversation_id
        or revision.live_message_id is None
        for revision in revisions.values()
    ):
        raise ValueError("revision_owner_mismatch")
    message_ids = tuple(
        cast(str, revisions[revision_id].live_message_id)
        for revision_id in revision_ids
    )
    message_placeholders = ",".join("?" for _ in message_ids)
    message_rows = cursor.execute(
        f"""SELECT id, conversation_id, parent_message_id, sender, role, content,
                   image_data, image_mime_type, provider_continuation_json,
                   thinking_blocks_json, assistant_generation_state
              FROM messages
             WHERE id IN ({message_placeholders})""",
        message_ids,
    ).fetchall()
    attachments: dict[str, list[dict[str, object]]] = {
        message_id: [] for message_id in message_ids
    }
    for row in cursor.execute(
        f"""SELECT message_id, position, data, mime_type, display_name
              FROM message_attachments
             WHERE message_id IN ({message_placeholders})
             ORDER BY message_id, position""",
        message_ids,
    ):
        attachments[str(row[0])].append(
            {
                "position": int(row[1]),
                "data_base64": base64.b64encode(bytes(row[2])).decode("ascii"),
                "mime_type": row[3],
                "display_name": row[4],
            }
        )
    envelopes: dict[str, dict[str, object]] = {}
    for row in message_rows:
        message_id = str(row[0])
        image_data = row[6]
        envelopes[message_id] = {
            "message_id": message_id,
            "conversation_id": row[1],
            "parent_message_id": row[2],
            "sender": row[3],
            "role": row[4],
            "content": row[5],
            "image_data_base64": (
                None
                if image_data is None
                else base64.b64encode(bytes(image_data)).decode("ascii")
            ),
            "image_mime_type": row[7],
            "provider_continuation": _decode_optional_json(row[8]),
            "thinking_blocks": _decode_optional_json(row[9]),
            "assistant_generation_state": row[10],
            "attachments": attachments[message_id],
        }
    result: dict[str, dict[str, object]] = {}
    for revision_id in revision_ids:
        revision = revisions[revision_id]
        envelope = envelopes.get(cast(str, revision.live_message_id))
        if (
            envelope is None
            or envelope["message_id"] != revision.source_message_id
            or envelope["conversation_id"] != expected_conversation_id
        ):
            raise ValueError("semantic_revision_locator_mismatch")
        result[revision_id] = _project_revision_envelope(revision, envelope)
    return result


def project_semantic_revision_provider_continuations(
    cursor: sqlite3.Cursor,
    *,
    revision_ids: tuple[str, ...],
    expected_conversation_id: str,
) -> dict[str, object]:
    """Batch-project canonical continuation sidecars owned by saved revisions."""

    if not revision_ids:
        return {}
    if len(revision_ids) > 256 or len(set(revision_ids)) != len(revision_ids):
        raise ValueError("semantic_revision_batch")
    placeholders = ",".join("?" for _ in revision_ids)
    rows = cursor.execute(
        f"""SELECT r.revision_id, r.source_conversation_id,
                   r.source_message_id, r.live_message_id,
                   m.id, m.conversation_id, m.provider_continuation_json
              FROM console_trace_semantic_revisions AS r
              JOIN messages AS m ON m.id = r.live_message_id
             WHERE r.revision_id IN ({placeholders})""",
        revision_ids,
    ).fetchall()
    by_id = {str(row[0]): row for row in rows}
    result: dict[str, object] = {}
    for revision_id in revision_ids:
        row = by_id.get(revision_id)
        if (
            row is None
            or row[1] != expected_conversation_id
            or row[2] != row[3]
            or row[3] != row[4]
            or row[5] != expected_conversation_id
        ):
            raise ValueError("revision_owner_mismatch")
        continuation = _decode_optional_json(row[6])
        if continuation is None:
            raise ValueError("semantic_continuation_unavailable")
        result[revision_id] = continuation
    return result


def _project_revision_envelope(
    revision: SemanticRevisionRecord,
    envelope: dict[str, object],
) -> dict[str, object]:
    content: object = envelope["content"]
    media: list[dict[str, object]] = []
    primary = envelope["image_data_base64"]
    if isinstance(primary, str):
        media.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{envelope['image_mime_type'] or 'image/png'};"
                        f"base64,{primary}"
                    )
                },
            }
        )
    attachments = envelope["attachments"]
    if not isinstance(attachments, list):
        raise ValueError("semantic_message_unavailable")
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ValueError("semantic_message_unavailable")
        media.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:{attachment.get('mime_type') or 'image/png'};"
                        f"base64,{attachment['data_base64']}"
                    )
                },
            }
        )
    if media:
        content = ([{"type": "text", "text": content}] if content else []) + media
    return {"role": revision.normalized_role, "content": content}


class TraceCredentialSanitizer(Protocol):
    """Minimal sanitizer seam used by the transaction coordinator."""

    def sanitize(self, value: object) -> CredentialSanitizationResult:
        """Return a credential-filtered result for trace persistence."""


@dataclass(frozen=True, slots=True)
class SemanticMutationResult:
    """Ledger identities produced, or no identities for an untracked no-op."""

    previous_revision_id: str | None
    current_revision_id: str | None
    replacement_id: str | None
    materialized_policy_ids: tuple[str, ...]
    deleted: bool


class SemanticRevisionCoordinator:
    """Preserve semantic history around caller-owned canonical mutations."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        repository: ConsoleTraceRepository | None = None,
        sanitizer: TraceCredentialSanitizer | None = None,
    ) -> None:
        self.db = db
        self.repository = repository or ConsoleTraceRepository()
        self._sanitizer = sanitizer or CredentialSanitizer()

    def ensure_current_revision(
        self,
        cursor: sqlite3.Cursor,
        *,
        message_id: str,
        creation_reason: str = "legacy_reference",
    ) -> SemanticRevisionRecord:
        """Return or lazily create digest-free metadata for a live message."""

        self._require_transaction(cursor)
        return self._ensure_current_revision(
            cursor,
            message_id,
            creation_reason=creation_reason,
            advance_epoch=True,
        )

    def _ensure_current_revision(
        self,
        cursor: sqlite3.Cursor,
        message_id: str,
        *,
        creation_reason: str,
        advance_epoch: bool,
    ) -> SemanticRevisionRecord:
        existing = self._current_revision(cursor, message_id)
        if existing is not None:
            return existing
        message = self._message_envelope(cursor, message_id)
        return self._create_initial_revision(
            cursor,
            message_id=message_id,
            envelope=message,
            creation_reason=creation_reason,
            advance_epoch=advance_epoch,
        )

    def _create_initial_revision(
        self,
        cursor: sqlite3.Cursor,
        *,
        message_id: str,
        envelope: dict[str, object],
        creation_reason: str,
        advance_epoch: bool,
    ) -> SemanticRevisionRecord:
        """Create revision zero from a captured pre-mutation envelope."""

        normalized_role = self._token(str(envelope["role"]), "message")
        reason = self._token(creation_reason, "legacy_reference")
        if advance_epoch:
            return self.repository.ensure_semantic_revision(
                cursor,
                source_conversation_id=str(envelope["conversation_id"]),
                source_message_id=message_id,
                revision_sequence=0,
                normalized_role=normalized_role,
                content_kind=self._content_kind(envelope),
                creation_reason=reason,
                live_message_id=message_id,
            )
        revision_id = new_opaque_id()
        cursor.execute(
            """INSERT INTO console_trace_semantic_revisions(
                   revision_id, source_conversation_id, source_message_id,
                   revision_sequence, normalized_role, content_kind,
                   creation_reason, live_message_id)
                 VALUES (?, ?, ?, 0, ?, ?, ?, ?)""",
            (
                revision_id,
                str(envelope["conversation_id"]),
                message_id,
                normalized_role,
                self._content_kind(envelope),
                reason,
                message_id,
            ),
        )
        created = self.repository.get_semantic_revision(cursor, revision_id)
        assert created is not None
        return created

    def mutate_message(
        self,
        cursor: sqlite3.Cursor,
        *,
        message_id: str,
        creation_reason: str,
        mutate: Callable[[sqlite3.Cursor], object] | None = None,
        hard_delete: bool = False,
    ) -> SemanticMutationResult:
        """Preserve policies, mutate once, and append lineage atomically.

        The caller owns an active write transaction, preferably opened with
        ``transaction(immediate=True)``. This method uses a nested savepoint for
        its atomic sub-boundary but never commits or rolls back the caller's
        transaction.
        """

        self._require_transaction(cursor)
        if hard_delete == (mutate is not None):
            raise ValueError("choose exactly one mutation operation")
        savepoint = f"semantic_mutation_{new_opaque_id().replace('-', '')}"
        cursor.execute(f"SAVEPOINT {savepoint}")
        try:
            result = self._mutate_message(
                cursor,
                message_id=message_id,
                creation_reason=creation_reason,
                mutate=mutate,
                hard_delete=hard_delete,
            )
        except BaseException:
            cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        cursor.execute(f"RELEASE SAVEPOINT {savepoint}")
        return result

    def _mutate_message(
        self,
        cursor: sqlite3.Cursor,
        *,
        message_id: str,
        creation_reason: str,
        mutate: Callable[[sqlite3.Cursor], object] | None,
        hard_delete: bool,
    ) -> SemanticMutationResult:
        """Run one semantic mutation inside the method-local savepoint."""

        current = self._current_revision(cursor, message_id)
        old_envelope = self._message_envelope(cursor, message_id)
        authorization = self.db._semantic_mutation_authorization_for_coordinator(
            cursor.connection
        )
        with authorization._authorize(
            message_id=message_id,
            operations=_MUTATION_OPERATIONS,
        ):
            if hard_delete:
                if current is None:
                    current = self._create_initial_revision(
                        cursor,
                        message_id=message_id,
                        envelope=old_envelope,
                        creation_reason="legacy_reference",
                        advance_epoch=False,
                    )
                policy_ids = self._reachable_policy_ids(cursor, current.revision_id)
                self._materialize_policies(
                    cursor,
                    revision_id=current.revision_id,
                    policy_ids=policy_ids,
                    envelope=old_envelope,
                )
                retired_at = self._retire_live_locator(cursor, current.revision_id)
                cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
                if cursor.rowcount != 1:
                    raise RuntimeError("semantic_message_delete_failed")
                new_envelope = None
            else:
                assert mutate is not None
                mutate(cursor)
                authorization._assert_current_transaction()
                new_envelope = self._message_envelope(cursor, message_id)
                if new_envelope == old_envelope:
                    revision_id = None if current is None else current.revision_id
                    return SemanticMutationResult(
                        previous_revision_id=revision_id,
                        current_revision_id=revision_id,
                        replacement_id=None,
                        materialized_policy_ids=(),
                        deleted=False,
                    )
                if current is None:
                    current = self._create_initial_revision(
                        cursor,
                        message_id=message_id,
                        envelope=old_envelope,
                        creation_reason="legacy_reference",
                        advance_epoch=False,
                    )
                policy_ids = self._reachable_policy_ids(cursor, current.revision_id)
                self._materialize_policies(
                    cursor,
                    revision_id=current.revision_id,
                    policy_ids=policy_ids,
                    envelope=old_envelope,
                )
                retired_at = self._retire_live_locator(cursor, current.revision_id)

        assert current is not None
        next_revision = self._append_successor_revision(
            cursor,
            current=current,
            creation_reason=creation_reason,
            envelope=new_envelope,
            retired_at=retired_at,
        )
        replacement_id = self._append_surface_replacement(
            cursor,
            previous_revision_id=current.revision_id,
            current_revision_id=next_revision.revision_id,
            hard_delete=hard_delete,
        )
        self._advance_epoch_once(cursor)
        return SemanticMutationResult(
            previous_revision_id=current.revision_id,
            current_revision_id=next_revision.revision_id,
            replacement_id=replacement_id,
            materialized_policy_ids=policy_ids,
            deleted=hard_delete,
        )

    @staticmethod
    def _require_transaction(cursor: sqlite3.Cursor) -> None:
        if not cursor.connection.in_transaction:
            raise RuntimeError("caller_transaction_required")

    def _current_revision(
        self, cursor: sqlite3.Cursor, message_id: str
    ) -> SemanticRevisionRecord | None:
        row = cursor.execute(
            """
            SELECT revision_id, source_conversation_id, source_message_id,
                   revision_sequence, normalized_role, content_kind,
                   creation_reason, predecessor_revision_id, live_message_id,
                   live_locator_retired_at
              FROM console_trace_semantic_revisions
             WHERE live_message_id = ?
            """,
            (message_id,),
        ).fetchone()
        return None if row is None else SemanticRevisionRecord(*row)

    @staticmethod
    def _message_envelope(cursor: sqlite3.Cursor, message_id: str) -> dict[str, object]:
        row = cursor.execute(
            """
            SELECT id, conversation_id, parent_message_id, sender, role, content,
                   image_data, image_mime_type, provider_continuation_json,
                   thinking_blocks_json, assistant_generation_state
              FROM messages
             WHERE id = ?
            """,
            (message_id,),
        ).fetchone()
        if row is None:
            raise ValueError("semantic_message_unavailable")
        attachments = [
            {
                "position": int(item[0]),
                "data_base64": base64.b64encode(bytes(item[1])).decode("ascii"),
                "mime_type": item[2],
                "display_name": item[3],
            }
            for item in cursor.execute(
                """
                SELECT position, data, mime_type, display_name
                  FROM message_attachments
                 WHERE message_id = ?
                 ORDER BY position
                """,
                (message_id,),
            )
        ]
        image_data = row[6]
        return {
            "message_id": row[0],
            "conversation_id": row[1],
            "parent_message_id": row[2],
            "sender": row[3],
            "role": row[4],
            "content": row[5],
            "image_data_base64": (
                None
                if image_data is None
                else base64.b64encode(bytes(image_data)).decode("ascii")
            ),
            "image_mime_type": row[7],
            "provider_continuation": _decode_optional_json(row[8]),
            "thinking_blocks": _decode_optional_json(row[9]),
            "assistant_generation_state": row[10],
            "attachments": attachments,
        }

    @staticmethod
    def _content_kind(envelope: dict[str, object]) -> str:
        if envelope["image_data_base64"] is not None or envelope["attachments"]:
            return "multimodal"
        if envelope["provider_continuation"] is not None:
            return "tool"
        return "text"

    @staticmethod
    def _token(value: str, fallback: str) -> str:
        token = _TOKEN_CHARS.sub("_", value.strip().lower()).strip("_")
        return (token or fallback)[:64].rstrip("_")

    @staticmethod
    def _reachable_policy_ids(
        cursor: sqlite3.Cursor, revision_id: str
    ) -> tuple[str, ...]:
        rows = cursor.execute(
            """
            WITH RECURSIVE revision_descendants(node_id) AS (
              SELECT node_id
                FROM console_trace_surface_nodes
               WHERE semantic_revision_id = ?
              UNION
              SELECT child.node_id
                FROM revision_descendants AS predecessor
                JOIN console_trace_surface_nodes AS child
                  ON child.predecessor_node_id = predecessor.node_id
            )
            SELECT DISTINCT call.policy_id
              FROM revision_descendants AS descendant
              JOIN console_trace_calls AS call
                ON call.surface_node_id = descendant.node_id
            UNION
            SELECT DISTINCT call.policy_id
              FROM console_trace_response_links AS link
              JOIN console_trace_calls AS call ON call.call_id = link.call_id
             WHERE link.semantic_revision_id = ?
            UNION
            SELECT policy_id
              FROM console_trace_revision_bindings
             WHERE revision_id = ?
            ORDER BY 1
            """,
            (revision_id, revision_id, revision_id),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def _materialize_policies(
        self,
        cursor: sqlite3.Cursor,
        *,
        revision_id: str,
        policy_ids: tuple[str, ...],
        envelope: dict[str, object],
    ) -> None:
        pending = []
        for policy_id in policy_ids:
            if (
                cursor.execute(
                    """
                    SELECT 1 FROM console_trace_revision_bindings
                     WHERE revision_id = ? AND policy_id = ?
                    """,
                    (revision_id, policy_id),
                ).fetchone()
                is not None
            ):
                continue
            policy = self.repository.get_policy(cursor, policy_id)
            if policy is None:
                raise RuntimeError("trace_policy_unavailable")
            pending.append(policy)
        if not pending:
            return
        supported = tuple(
            policy
            for policy in pending
            if policy.credential_filter_version == CREDENTIAL_FILTER_VERSION
            and not policy.pii_redaction_enabled
            and policy.pii_ruleset_revision_id is None
        )
        for policy in pending:
            if policy in supported:
                continue
            cursor.execute(
                """
                INSERT INTO console_trace_revision_bindings(
                  revision_id, policy_id, binding_outcome,
                  omission_reason_code
                ) VALUES (?, ?, 'omission', ?)
                """,
                (
                    revision_id,
                    policy.policy_id,
                    _TRACE_REDACTION_POLICY_UNSUPPORTED,
                ),
            )
        if not supported:
            return
        provider_envelope = {
            key: value
            for key, value in envelope.items()
            if key not in _INTERNAL_ENVELOPE_KEYS
        }
        result = self._sanitizer.sanitize(provider_envelope)
        artifact_id: str | None = None
        detector_matches = result.detector_version == CREDENTIAL_FILTER_VERSION
        if result.available and detector_matches:
            sanitized_bytes = json.dumps(
                result.value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            artifact = self.repository.store_sanitized_artifact(
                cursor,
                sanitized_bytes=sanitized_bytes,
                media_type="application/vnd.tldw.semantic-message+json",
                normalization_version="semantic-envelope-v1",
            )
            artifact_id = artifact.artifact_id
        for policy in supported:
            if artifact_id is not None:
                cursor.execute(
                    """
                    INSERT INTO console_trace_revision_bindings(
                      revision_id, policy_id, binding_outcome, artifact_id
                    ) VALUES (?, ?, 'artifact', ?)
                    """,
                    (revision_id, policy.policy_id, artifact_id),
                )
            else:
                omission_reason = (
                    result.omission_reason_code
                    if detector_matches
                    else _TRACE_REDACTION_POLICY_UNSUPPORTED
                )
                if omission_reason is None:
                    omission_reason = _TRACE_REDACTION_POLICY_UNSUPPORTED
                cursor.execute(
                    """
                    INSERT INTO console_trace_revision_bindings(
                      revision_id, policy_id, binding_outcome,
                      omission_reason_code
                    ) VALUES (?, ?, 'omission', ?)
                    """,
                    (revision_id, policy.policy_id, omission_reason),
                )

    @staticmethod
    def _retire_live_locator(cursor: sqlite3.Cursor, revision_id: str) -> str:
        retired_at = str(
            cursor.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0]
        )
        update = cursor.execute(
            """
            UPDATE console_trace_semantic_revisions
               SET live_message_id = NULL, live_locator_retired_at = ?
             WHERE revision_id = ? AND live_message_id IS NOT NULL
            """,
            (retired_at, revision_id),
        )
        if update.rowcount != 1:
            raise RuntimeError("semantic_revision_locator_retirement_failed")
        return retired_at

    def _append_successor_revision(
        self,
        cursor: sqlite3.Cursor,
        *,
        current: SemanticRevisionRecord,
        creation_reason: str,
        envelope: dict[str, object] | None,
        retired_at: str,
    ) -> SemanticRevisionRecord:
        revision_id = new_opaque_id()
        if envelope is None:
            role = current.normalized_role
            content_kind = "deleted"
            live_message_id = None
            locator_retired_at = retired_at
        else:
            role = self._token(str(envelope["role"]), "message")
            content_kind = self._content_kind(envelope)
            live_message_id = current.source_message_id
            locator_retired_at = None
        cursor.execute(
            """
            INSERT INTO console_trace_semantic_revisions(
              revision_id, source_conversation_id, source_message_id,
              revision_sequence, normalized_role, content_kind,
              creation_reason, predecessor_revision_id, live_message_id,
              live_locator_retired_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                current.source_conversation_id,
                current.source_message_id,
                current.revision_sequence + 1,
                role,
                content_kind,
                self._token(creation_reason, "semantic_mutation"),
                current.revision_id,
                live_message_id,
                locator_retired_at,
            ),
        )
        created = self.repository.get_semantic_revision(cursor, revision_id)
        assert created is not None
        return created

    @staticmethod
    def _append_surface_replacement(
        cursor: sqlite3.Cursor,
        *,
        previous_revision_id: str,
        current_revision_id: str,
        hard_delete: bool,
    ) -> str | None:
        old = cursor.execute(
            """
            SELECT node.node_id, node.segment_id, node.sequence
              FROM console_trace_surface_nodes AS node
              JOIN console_trace_owners AS owner
                ON owner.root_segment_id = node.segment_id
             WHERE node.semantic_revision_id = ? AND owner.attached = 1
             ORDER BY node.created_at DESC, node.node_id DESC
             LIMIT 1
            """,
            (previous_revision_id,),
        ).fetchone()
        if old is None:
            return None
        head = cursor.execute(
            """
            SELECT node_id, sequence
              FROM console_trace_surface_nodes
             WHERE segment_id = ?
             ORDER BY sequence DESC
             LIMIT 1
            """,
            (old[1],),
        ).fetchone()
        assert head is not None
        replacement_node_id = new_opaque_id()
        next_node_sequence = int(head[1]) + 1
        if hard_delete:
            cursor.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                  node_id, segment_id, sequence, predecessor_node_id,
                  component_kind, reference_kind, omission_reason_code
                ) VALUES (?, ?, ?, ?, 'message', 'omission', 'message_deleted')
                """,
                (replacement_node_id, old[1], next_node_sequence, head[0]),
            )
        else:
            cursor.execute(
                """
                INSERT INTO console_trace_surface_nodes(
                  node_id, segment_id, sequence, predecessor_node_id,
                  component_kind, reference_kind, semantic_revision_id
                ) VALUES (?, ?, ?, ?, 'message', 'revision', ?)
                """,
                (
                    replacement_node_id,
                    old[1],
                    next_node_sequence,
                    head[0],
                    current_revision_id,
                ),
            )
        replacement_id = new_opaque_id()
        cursor.execute(
            """
            INSERT INTO console_trace_surface_replacements(
              replacement_id, segment_id, predecessor_head_id,
              start_node_id, start_sequence, end_node_id, end_sequence,
              replacement_node_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                replacement_id,
                old[1],
                head[0],
                old[0],
                old[2],
                old[0],
                old[2],
                replacement_node_id,
            ),
        )
        event_sequence = int(
            cursor.execute(
                """
                SELECT COALESCE(MAX(sequence), -1) + 1
                  FROM console_trace_events WHERE segment_id = ?
                """,
                (old[1],),
            ).fetchone()[0]
        )
        cursor.execute(
            """
            INSERT INTO console_trace_events(
              event_id, segment_id, sequence, event_type,
              surface_replacement_id
            ) VALUES (?, ?, ?, 'surface_replace', ?)
            """,
            (new_opaque_id(), old[1], event_sequence, replacement_id),
        )
        return replacement_id

    @staticmethod
    def _advance_epoch_once(cursor: sqlite3.Cursor) -> None:
        update = cursor.execute(
            """
            UPDATE console_trace_graph_epoch
               SET epoch = epoch + 1, updated_at = CURRENT_TIMESTAMP
             WHERE singleton_id = 1
            """
        )
        if update.rowcount != 1:
            raise RuntimeError("graph_epoch_unavailable")


def _decode_optional_json(value: object) -> object:
    """Decode stored JSON without accepting invalid durable envelopes."""

    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("semantic_envelope_json_invalid")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("semantic_envelope_json_invalid") from exc
