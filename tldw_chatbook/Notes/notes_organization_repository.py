"""SQLite projection for the portable Notes organization Sync domains."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Sync_Interop.notes_organization import (
    parse_notes_organization_payload,
    validate_organization_object_id,
)

_RESOURCE_TABLES = {
    "notes.keyword": ("keywords", "keyword"),
    "notes.keyword_collection": ("keyword_collections", "name"),
    "notes.folder": ("note_folders", "name"),
}
_HEX_HASH = re.compile(r"[0-9a-f]{64}")
_PORTABLE_PATH_LIMIT = 500
LOCAL_ID_ALLOCATION_ATTEMPTS = 8


class NotesOrganizationRepositoryError(ValueError):
    """Reject an organization projection that cannot be represented safely."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of applying one validated server envelope."""

    status: Literal["applied", "blocked", "duplicate", "stale"]
    local_id: str | int | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class _IntentLineage:
    """Current exact predecessor/base and next local version for one object."""

    predecessor_intent_id: str | None
    predecessor_operation: str | None
    base_server_cursor: str | None
    base_object_revision: int | None
    base_object_hash: str | None
    next_source_version: int


def portable_collision_key(name: str, *, maximum: int = 500) -> str:
    """Return the server-compatible casefold key for one portable segment."""

    if not isinstance(name, str):
        raise NotesOrganizationRepositoryError("invalid_name", "name must be text")
    display = name.strip()
    if (
        not display
        or display in {".", ".."}
        or len(display) > maximum
        or "/" in display
        or "\\" in display
        or "\x00" in display
    ):
        raise NotesOrganizationRepositoryError(
            "invalid_name", "name is not a portable path segment"
        )
    key = display.casefold()
    if not key or key in {".", ".."} or "/" in key or "\\" in key:
        raise NotesOrganizationRepositoryError(
            "invalid_name", "name is not a portable path segment"
        )
    return key


def portable_relative_path(segments: Sequence[str]) -> str:
    """Build a bounded relative portable path from validated segments."""

    if isinstance(segments, (str, bytes)) or not segments:
        raise NotesOrganizationRepositoryError(
            "invalid_path", "portable path requires at least one segment"
        )
    path = "/".join(portable_collision_key(segment) for segment in segments)
    if len(path) > _PORTABLE_PATH_LIMIT:
        raise NotesOrganizationRepositoryError(
            "invalid_path", "portable relative path exceeds 500 characters"
        )
    return path


class NotesOrganizationRepository:
    """Materialize Notes organization state using a caller-owned transaction."""

    def __init__(
        self, db: CharactersRAGDB, *, server_profile_id: str = "default"
    ) -> None:
        if not isinstance(db, CharactersRAGDB):
            raise TypeError("db must be a CharactersRAGDB instance")
        if not isinstance(server_profile_id, str) or not server_profile_id.strip():
            raise ValueError("server_profile_id must be non-blank text")
        self.db = db
        self.server_profile_id = server_profile_id.strip()

    def get_resource_by_sync_id(
        self, domain: str, sync_id: str, *, cursor: sqlite3.Cursor | None = None
    ) -> sqlite3.Row | None:
        """Look up one local resource by its portable UUID."""

        table_column = _RESOURCE_TABLES.get(domain)
        if table_column is None:
            raise NotesOrganizationRepositoryError(
                "invalid_domain", "domain is not an organization resource"
            )
        connection = cursor if cursor is not None else self.db.get_connection()
        return connection.execute(
            f"SELECT * FROM {table_column[0]} WHERE sync_id = ?", (sync_id,)
        ).fetchone()

    def apply_envelope(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        domain: str,
        object_id: str,
        operation: str,
        payload: Mapping[str, object],
        object_revision: int,
        object_hash: str,
        server_cursor: str,
        base_server_cursor: str | None = None,
        base_object_revision: int | None = None,
        base_object_hash: str | None = None,
        restore_intent: bool = False,
    ) -> ApplyResult:
        """Validate and apply one envelope without opening a nested transaction."""

        self._require_owned_cursor(cursor)
        self._validate_apply_metadata(
            cursor,
            dataset_id,
            object_revision,
            object_hash,
            server_cursor,
            restore_intent,
        )
        normalized = parse_notes_organization_payload(domain, operation, payload)
        validate_organization_object_id(domain, object_id, normalized)
        agent_lessons_candidate = None
        is_exact_agent_lessons_root = (
            domain == "notes.folder"
            and operation == "upsert"
            and normalized.get("name") == "Agent_Lessons"
            and normalized.get("parent_sync_id") is None
        )
        if is_exact_agent_lessons_root:
            agent_lessons_candidate = self._agent_lessons_local_seed_candidate(
                cursor, dataset_id=dataset_id, remote_object_id=object_id
            )
            from tldw_chatbook.Notes.agent_lessons import (
                record_remote_agent_lessons_seed_evidence,
            )

            record_remote_agent_lessons_seed_evidence(
                cursor,
                profile_id=self.server_profile_id,
                dataset_id=dataset_id,
                folder_sync_id=object_id,
            )
        payload_json = _canonical_json(normalized)
        payload_hash = _sha256(payload_json)
        head = cursor.execute(
            "SELECT operation, object_revision, object_hash, server_cursor, apply_state FROM "
            "notes_organization_heads WHERE server_profile_id = ? AND dataset_id = ? "
            "AND domain = ? AND object_id = ?",
            (self.server_profile_id, dataset_id, domain, object_id),
        ).fetchone()
        if restore_intent and (
            operation != "upsert"
            or head is None
            or str(head["operation"]) != "tombstone"
            or str(head["apply_state"]) != "applied"
            or base_object_revision is None
            or base_object_hash is None
            or base_object_revision != int(head["object_revision"])
            or base_object_hash != str(head["object_hash"])
            or (
                base_server_cursor is not None
                and base_server_cursor != str(head["server_cursor"])
            )
        ):
            return ApplyResult("blocked", reason_code="restore_intent_invalid")
        is_restore = False
        if head is not None:
            current_revision = int(head["object_revision"])
            if object_revision < current_revision:
                if agent_lessons_candidate is not None:
                    self._record_agent_lessons_seed_review(
                        cursor,
                        dataset_id=dataset_id,
                        remote_object_id=object_id,
                        candidate=agent_lessons_candidate,
                    )
                return ApplyResult("stale")
            if object_revision == current_revision:
                if str(head["object_hash"]) != object_hash:
                    raise NotesOrganizationRepositoryError(
                        "revision_hash_conflict",
                        "one object revision cannot have multiple hashes",
                    )
                if str(head["apply_state"]) == "applied":
                    if agent_lessons_candidate is not None:
                        self._record_agent_lessons_seed_review(
                            cursor,
                            dataset_id=dataset_id,
                            remote_object_id=object_id,
                            candidate=agent_lessons_candidate,
                        )
                    return ApplyResult("duplicate")
            is_restore = str(head["operation"]) == "tombstone" and operation == "upsert"
            if is_restore and not restore_intent:
                # Preserve the tombstone head so an identical replay cannot erase
                # the restore requirement merely by being seen once.
                return ApplyResult("blocked", reason_code="restore_intent_required")

        if agent_lessons_candidate is not None:
            retired = self._retire_untouched_agent_lessons_seed(
                cursor,
                dataset_id=dataset_id,
                remote_object_id=object_id,
                candidate=agent_lessons_candidate,
            )
            if not retired:
                self._record_agent_lessons_seed_review(
                    cursor,
                    dataset_id=dataset_id,
                    remote_object_id=object_id,
                    candidate=agent_lessons_candidate,
                )

        try:
            local_id = self._materialize(
                cursor,
                dataset_id=dataset_id,
                domain=domain,
                object_id=object_id,
                operation=operation,
                payload=normalized,
            )
        except NotesOrganizationRepositoryError as exc:
            if is_restore:
                # A failed restore must leave the accepted tombstone as the
                # canonical head so later retries still require explicit intent.
                return ApplyResult("blocked", reason_code=exc.reason_code)
            if (
                operation == "tombstone"
                and domain in _RESOURCE_TABLES
                and exc.reason_code == "missing_dependency"
            ):
                # The resource's earlier upsert may still be in flight. Keeping
                # this tombstone out of the canonical head lets that history
                # materialize before the same tombstone revision retries.
                return ApplyResult("blocked", reason_code=exc.reason_code)
            if exc.reason_code == "local_representation_collision":
                # The durable adoption review is the conflict record. Do not
                # advance the canonical object head until that review resolves.
                return ApplyResult("blocked", reason_code=exc.reason_code)
            self._write_head(
                cursor,
                dataset_id=dataset_id,
                domain=domain,
                object_id=object_id,
                operation=operation,
                payload_json=payload_json,
                payload_hash=payload_hash,
                object_revision=object_revision,
                object_hash=object_hash,
                server_cursor=server_cursor,
                apply_state="blocked",
                reason_code=exc.reason_code,
            )
            return ApplyResult("blocked", reason_code=exc.reason_code)
        return self._write_head(
            cursor,
            dataset_id=dataset_id,
            domain=domain,
            object_id=object_id,
            operation=operation,
            payload_json=payload_json,
            payload_hash=payload_hash,
            object_revision=object_revision,
            object_hash=object_hash,
            server_cursor=server_cursor,
            apply_state="applied",
            local_id=local_id,
        )

    def _agent_lessons_local_seed_candidate(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        remote_object_id: str,
    ) -> sqlite3.Row | None:
        """Capture the pre-evidence local candidate for later race handling."""

        state = cursor.execute(
            "SELECT * FROM agent_lessons_seed_state WHERE profile_id = ? "
            "AND dataset_id = ? AND scope_mode = 'synchronized' AND state = 'seeded'",
            (self.server_profile_id, dataset_id),
        ).fetchone()
        if (
            state is None
            or not state["folder_sync_id"]
            or str(state["folder_sync_id"]) == remote_object_id
        ):
            return None
        return state

    def _retire_untouched_agent_lessons_seed(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        remote_object_id: str,
        candidate: sqlite3.Row,
    ) -> bool:
        """Retire only a provably untouched unpublished local seed candidate."""

        from tldw_chatbook.Notes.agent_lessons import agent_lessons_seed_fingerprint

        local_sync_id = str(candidate["folder_sync_id"])
        if local_sync_id == remote_object_id:
            return False
        expected = agent_lessons_seed_fingerprint(
            category="coordinator_created",
            profile_id=self.server_profile_id,
            dataset_id=dataset_id,
            folder_sync_id=local_sync_id,
        )
        if str(candidate["seed_fingerprint"]) != expected:
            return False
        folder = cursor.execute(
            "SELECT * FROM note_folders WHERE sync_id = ? AND parent_id IS NULL "
            "AND name = 'Agent_Lessons' COLLATE BINARY AND deleted = 0 AND version = 1",
            (local_sync_id,),
        ).fetchone()
        if folder is None:
            return False
        folder_id = str(folder["id"])
        if cursor.execute(
            "SELECT 1 FROM note_folders WHERE parent_id = ? LIMIT 1", (folder_id,)
        ).fetchone() is not None:
            return False
        if cursor.execute(
            "SELECT 1 FROM note_folder_memberships WHERE folder_id = ? LIMIT 1",
            (folder_id,),
        ).fetchone() is not None:
            return False
        if cursor.execute(
            "SELECT 1 FROM note_folder_sync_suppressions WHERE folder_sync_id = ? LIMIT 1",
            (local_sync_id,),
        ).fetchone() is not None:
            return False
        if cursor.execute(
            "SELECT 1 FROM notes_organization_heads WHERE server_profile_id = ? "
            "AND dataset_id = ? AND object_id = ? LIMIT 1",
            (self.server_profile_id, dataset_id, local_sync_id),
        ).fetchone() is not None:
            return False
        if cursor.execute(
            "SELECT 1 FROM notes_organization_adoption_reviews WHERE "
            "server_profile_id = ? AND dataset_id = ? AND (local_object_id = ? "
            "OR remote_object_id = ?) LIMIT 1",
            (self.server_profile_id, dataset_id, folder_id, local_sync_id),
        ).fetchone() is not None:
            return False
        if cursor.execute(
            "SELECT 1 FROM note_organization_receipts WHERE "
            "requested_folder_sync_id = ? OR requested_folder_name = 'Agent_Lessons' "
            "COLLATE BINARY LIMIT 1",
            (local_sync_id,),
        ).fetchone() is not None:
            return False

        intents = cursor.execute(
            "SELECT * FROM notes_organization_sync_intents WHERE server_profile_id = ? "
            "AND dataset_id = ?",
            (self.server_profile_id, dataset_id),
        ).fetchall()
        candidate = None
        for intent in intents:
            mentions_candidate = str(intent["object_id"]) == local_sync_id
            if not mentions_candidate:
                try:
                    mentions_candidate = local_sync_id in _json_scalar_values(
                        json.loads(str(intent["payload_json"]))
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    return False
            if not mentions_candidate:
                return False
            if candidate is not None:
                return False
            candidate = intent
        if (
            candidate is None
            or candidate["domain"] != "notes.folder"
            or candidate["operation"] != "upsert"
            or int(candidate["source_version"]) != 1
            or candidate["outbox_client_envelope_id"] is not None
            or candidate["copied_at"] is not None
            or candidate["acknowledged_at"] is not None
            or json.loads(str(candidate["payload_json"]))
            != {"name": "Agent_Lessons", "parent_sync_id": None}
        ):
            return False

        cursor.execute(
            "DELETE FROM notes_organization_sync_intents WHERE intent_id = ?",
            (str(candidate["intent_id"]),),
        )
        cursor.execute(
            "UPDATE note_folders SET deleted = 1 WHERE id = ? AND version = 1 "
            "AND deleted = 0",
            (folder_id,),
        )
        return cursor.rowcount == 1

    def _record_agent_lessons_seed_review(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        remote_object_id: str,
        candidate: sqlite3.Row,
    ) -> None:
        """Hold any non-retirable local seed identity for explicit adoption."""

        local_sync_id = str(candidate["folder_sync_id"])
        local = cursor.execute(
            "SELECT id, name, normalized_path FROM note_folders WHERE sync_id = ?",
            (local_sync_id,),
        ).fetchone()
        if local is None:
            return
        self._record_adoption_review(
            cursor,
            dataset_id,
            domain="notes.folder",
            local_object_id=str(local["id"]),
            remote_object_id=remote_object_id,
            collision_key="agent_lessons",
            portable_path=(
                str(local["normalized_path"])
                if local["normalized_path"] is not None
                else "agent_lessons"
            ),
            display=str(local["name"]),
        )

    def _record_intent_with_cursor(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile: str,
        dataset: str,
        domain: str,
        object_id: str,
        operation: str,
        payload: Mapping[str, object],
        routing_metadata: Mapping[str, object] | None = None,
        source_version: int,
        _lineage: _IntentLineage | None = None,
    ) -> str:
        """Persist one immutable canonical intent and return its stable identity."""

        self._require_owned_cursor(cursor)
        if (
            not isinstance(source_version, int)
            or isinstance(source_version, bool)
            or source_version < 1
        ):
            raise ValueError("source_version must be a positive integer")
        if (
            not isinstance(profile, str)
            or not profile.strip()
            or not isinstance(dataset, str)
            or not dataset.strip()
        ):
            raise ValueError("profile and dataset must be non-blank text")
        normalized = parse_notes_organization_payload(domain, operation, payload)
        validate_organization_object_id(domain, object_id, normalized)
        payload_json = _canonical_json(normalized)
        payload_hash = _sha256(payload_json)
        if routing_metadata is None:
            normalized_routing: dict[str, object] = {}
        elif not isinstance(routing_metadata, Mapping) or any(
            not isinstance(key, str) for key in routing_metadata
        ):
            raise ValueError("routing_metadata must be a JSON object")
        else:
            normalized_routing = dict(routing_metadata)
        restore_intent = normalized_routing.get("restore_intent")
        if (restore_intent is not None and restore_intent is not True) or (
            restore_intent is True and operation != "upsert"
        ):
            raise ValueError("restore_intent must be literal true on an upsert")
        try:
            routing_metadata_json = _canonical_json(normalized_routing)
        except (TypeError, ValueError) as exc:
            raise ValueError("routing_metadata must be a JSON object") from exc
        existing = cursor.execute(
            "SELECT intent_id, payload_hash, payload_json, routing_metadata_json "
            "FROM notes_organization_sync_intents "
            "WHERE server_profile_id = ? AND dataset_id = ? AND domain = ? "
            "AND object_id = ? AND source_version = ? AND operation = ?",
            (
                profile.strip(),
                dataset.strip(),
                domain,
                object_id,
                source_version,
                operation,
            ),
        ).fetchone()
        if existing is not None:
            if (
                existing["payload_hash"] != payload_hash
                or existing["payload_json"] != payload_json
                or existing["routing_metadata_json"] != routing_metadata_json
            ):
                raise NotesOrganizationRepositoryError(
                    "immutable_intent_conflict",
                    "an existing intent has different content",
                )
            return str(existing["intent_id"])

        lineage = _lineage or self._intent_lineage_with_cursor(
            cursor,
            profile=profile,
            dataset=dataset,
            domain=domain,
            object_id=object_id,
        )
        if restore_intent is True and lineage.predecessor_operation != "tombstone":
            raise ValueError("restore_intent requires the current tombstone base")

        intent_sequence = int(
            cursor.execute(
                "SELECT COALESCE(MAX(intent_sequence), 0) + 1 "
                "FROM notes_organization_sync_intents "
                "WHERE server_profile_id = ? AND dataset_id = ?",
                (profile.strip(), dataset.strip()),
            ).fetchone()[0]
        )

        identity = _canonical_json(
            {
                "profile": profile.strip(),
                "dataset": dataset.strip(),
                "domain": domain,
                "object_id": object_id,
                "operation": operation,
                "source_version": source_version,
            }
        )
        intent_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"tldw:notes-organization:{identity}")
        )
        dependencies = _dependency_refs(domain, normalized)
        now = _utc_timestamp()
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_intents(
                intent_id, intent_sequence, predecessor_intent_id,
                server_profile_id, dataset_id, domain, object_id, operation,
                schema_version, encryption_policy, payload_json, payload_hash,
                routing_metadata_json,
                base_server_cursor, base_object_revision, base_object_hash,
                dependency_refs_json, source_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 'server_trusted_v1', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                intent_sequence,
                lineage.predecessor_intent_id,
                profile.strip(),
                dataset.strip(),
                domain,
                object_id,
                operation,
                payload_json,
                payload_hash,
                routing_metadata_json,
                lineage.base_server_cursor,
                lineage.base_object_revision,
                lineage.base_object_hash,
                _canonical_json(dependencies),
                source_version,
                now,
            ),
        )
        return intent_id

    def _intent_lineage_with_cursor(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile: str,
        dataset: str,
        domain: str,
        object_id: str,
    ) -> _IntentLineage:
        """Inspect exact local predecessor, acknowledged head, and next version."""

        self._require_owned_cursor(cursor)
        normalized_profile = profile.strip()
        normalized_dataset = dataset.strip()
        predecessor = cursor.execute(
            "SELECT intent_id, operation, acknowledged_at "
            "FROM notes_organization_sync_intents WHERE server_profile_id = ? "
            "AND dataset_id = ? AND domain = ? AND object_id = ? "
            "ORDER BY intent_sequence DESC LIMIT 1",
            (
                normalized_profile,
                normalized_dataset,
                domain,
                object_id,
            ),
        ).fetchone()
        predecessor_intent_id = (
            str(predecessor["intent_id"]) if predecessor is not None else None
        )
        predecessor_operation: str | None
        if predecessor is not None and predecessor["acknowledged_at"] is None:
            base_server_cursor = None
            base_object_revision = None
            base_object_hash = None
            predecessor_operation = str(predecessor["operation"])
        else:
            head = cursor.execute(
                "SELECT operation, server_cursor, object_revision, object_hash "
                "FROM notes_organization_heads WHERE server_profile_id = ? "
                "AND dataset_id = ? AND domain = ? AND object_id = ?",
                (normalized_profile, normalized_dataset, domain, object_id),
            ).fetchone()
            base_server_cursor = (
                str(head["server_cursor"]) if head is not None else None
            )
            base_object_revision = (
                int(head["object_revision"]) if head is not None else None
            )
            base_object_hash = str(head["object_hash"]) if head is not None else None
            predecessor_operation = str(head["operation"]) if head is not None else None
        next_source_version = int(
            cursor.execute(
                "SELECT COALESCE(MAX(source_version), 0) + 1 "
                "FROM notes_organization_sync_intents "
                "WHERE server_profile_id = ? AND dataset_id = ? "
                "AND domain = ? AND object_id = ?",
                (normalized_profile, normalized_dataset, domain, object_id),
            ).fetchone()[0]
        )
        return _IntentLineage(
            predecessor_intent_id=predecessor_intent_id,
            predecessor_operation=predecessor_operation,
            base_server_cursor=base_server_cursor,
            base_object_revision=base_object_revision,
            base_object_hash=base_object_hash,
            next_source_version=next_source_version,
        )

    def _record_inferred_intent_with_cursor(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile: str,
        dataset: str,
        domain: str,
        object_id: str,
        operation: str,
        payload: Mapping[str, object],
        source_version: int | None = None,
    ) -> str:
        """Record a local mutation with exact restore routing derived from lineage."""

        lineage = self._intent_lineage_with_cursor(
            cursor,
            profile=profile,
            dataset=dataset,
            domain=domain,
            object_id=object_id,
        )
        routing_metadata = (
            {"restore_intent": True}
            if operation == "upsert" and lineage.predecessor_operation == "tombstone"
            else None
        )
        return self._record_intent_with_cursor(
            cursor,
            profile=profile,
            dataset=dataset,
            domain=domain,
            object_id=object_id,
            operation=operation,
            payload=payload,
            routing_metadata=routing_metadata,
            source_version=(
                lineage.next_source_version
                if source_version is None
                else source_version
            ),
            _lineage=lineage,
        )

    def record_intent(
        self,
        cursor: sqlite3.Cursor,
        *,
        profile: str,
        dataset: str,
        domain: str,
        object_id: str,
        operation: str,
        payload: Mapping[str, object],
        routing_metadata: Mapping[str, object] | None = None,
        source_version: int,
    ) -> str:
        """Persist one immutable intent through a caller-owned cursor."""

        return self._record_intent_with_cursor(
            cursor,
            profile=profile,
            dataset=dataset,
            domain=domain,
            object_id=object_id,
            operation=operation,
            payload=payload,
            routing_metadata=routing_metadata,
            source_version=source_version,
        )

    def advance_inventory_checkpoint(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        expected_phase: str,
        expected_key: str | None,
        inventory_phase: str,
        last_inventory_key: str | None,
    ) -> None:
        """Compare-and-swap one legacy-inventory checkpoint in an owned transaction."""

        self._require_owned_cursor(cursor)
        result = cursor.execute(
            """
            UPDATE notes_organization_sync_checkpoints
               SET inventory_phase = ?, last_inventory_key = ?, updated_at = ?
             WHERE server_profile_id = ? AND dataset_id = ?
               AND inventory_phase = ? AND last_inventory_key IS ?
            """,
            (
                inventory_phase,
                last_inventory_key,
                _utc_timestamp(),
                self.server_profile_id,
                dataset_id,
                expected_phase,
                expected_key,
            ),
        )
        if result.rowcount != 1:
            raise NotesOrganizationRepositoryError(
                "stale_inventory_checkpoint",
                "legacy inventory checkpoint changed concurrently",
            )

    def apply_resolved_inventory_merge(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        domain: str,
        local_object_id: str,
        remote_object_id: str,
    ) -> None:
        """Assign the explicitly reviewed remote identity to one legacy resource."""

        self._require_owned_cursor(cursor)
        table_column = _RESOURCE_TABLES.get(domain)
        if table_column is None:
            raise NotesOrganizationRepositoryError(
                "invalid_domain", "merge target is not an organization resource"
            )
        review = cursor.execute(
            """
            SELECT 1 FROM notes_organization_adoption_reviews
             WHERE server_profile_id = ? AND dataset_id = ? AND domain = ?
               AND local_object_id = ? AND remote_object_id = ?
               AND state = 'resolved' AND resolution = 'merge'
            """,
            (
                self.server_profile_id,
                dataset_id,
                domain,
                local_object_id,
                remote_object_id,
            ),
        ).fetchone()
        if review is None:
            raise NotesOrganizationRepositoryError(
                "adoption_review_required", "resource merge was not explicitly reviewed"
            )
        table = table_column[0]
        collision = cursor.execute(
            f"SELECT id FROM {table} WHERE sync_id = ? AND CAST(id AS TEXT) <> ?",
            (remote_object_id, local_object_id),
        ).fetchone()
        if collision is not None:
            raise NotesOrganizationRepositoryError(
                "adoption_identity_collision",
                "reviewed remote identity is already assigned locally",
            )
        resource = cursor.execute(
            f"SELECT sync_id FROM {table} WHERE CAST(id AS TEXT) = ?",
            (local_object_id,),
        ).fetchone()
        if resource is None:
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "reviewed merge target no longer exists"
            )
        if domain == "notes.folder" and resource["sync_id"] != remote_object_id:
            cursor.execute(
                "UPDATE OR IGNORE note_folder_sync_suppressions "
                "SET folder_sync_id = ? WHERE folder_sync_id = ?",
                (remote_object_id, resource["sync_id"]),
            )
            cursor.execute(
                "DELETE FROM note_folder_sync_suppressions WHERE folder_sync_id = ?",
                (resource["sync_id"],),
            )
        result = cursor.execute(
            f"UPDATE {table} SET sync_id = ? WHERE CAST(id AS TEXT) = ?",
            (remote_object_id, local_object_id),
        )
        if result.rowcount != 1:
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "reviewed merge target no longer exists"
            )

    def effective_folder_sync_ids(
        self, note_id: str, *, cursor: sqlite3.Cursor | None = None
    ) -> tuple[str, ...]:
        """Return active placements after applying portable suppressions."""

        connection = cursor if cursor is not None else self.db.get_connection()
        rows = connection.execute(
            """
            SELECT DISTINCT f.sync_id
              FROM note_folder_memberships AS m
              JOIN note_folders AS f ON f.id = m.folder_id
             WHERE m.note_id = ? AND m.deleted = 0 AND f.deleted = 0
               AND (m.ownership = 'manual' OR m.owner_active = 1)
               AND f.sync_id IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM note_folder_sync_suppressions AS s
                    WHERE s.note_id = m.note_id AND s.folder_sync_id = f.sync_id
               )
               AND NOT EXISTS (
                   WITH RECURSIVE ancestors(id, parent_id, deleted) AS (
                       SELECT id, parent_id, deleted FROM note_folders
                        WHERE id = f.parent_id
                       UNION ALL
                       SELECT parent.id, parent.parent_id, parent.deleted
                         FROM note_folders AS parent
                         JOIN ancestors ON parent.id = ancestors.parent_id
                   )
                   SELECT 1 FROM ancestors WHERE deleted = 1
               )
             ORDER BY f.sync_id
            """,
            (note_id,),
        ).fetchall()
        return tuple(str(row["sync_id"]) for row in rows)

    @staticmethod
    def _validate_apply_metadata(
        cursor: sqlite3.Cursor,
        dataset_id: str,
        revision: int,
        object_hash: str,
        server_cursor: str,
        restore_intent: bool,
    ) -> None:
        if not isinstance(cursor, sqlite3.Cursor):
            raise TypeError("cursor must be a sqlite3.Cursor")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset_id must be non-blank text")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ValueError("object_revision must be a positive integer")
        if not isinstance(object_hash, str) or _HEX_HASH.fullmatch(object_hash) is None:
            raise ValueError("object_hash must be a lowercase SHA-256 digest")
        if not isinstance(server_cursor, str) or not server_cursor:
            raise ValueError("server_cursor must be non-blank text")
        if not isinstance(restore_intent, bool):
            raise ValueError("restore_intent must be a boolean")

    def _materialize(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        domain: str,
        object_id: str,
        operation: str,
        payload: Mapping[str, object],
    ) -> str | int | None:
        if domain == "notes.keyword":
            return self._materialize_keyword(
                cursor, dataset_id, object_id, operation, payload
            )
        if domain == "notes.keyword_collection":
            return self._materialize_collection(
                cursor, dataset_id, object_id, operation, payload
            )
        if domain == "notes.folder":
            return self._materialize_folder(
                cursor, dataset_id, object_id, operation, payload
            )
        if domain == "notes.keyword_link":
            self._materialize_keyword_link(cursor, operation, payload)
            return None
        if domain == "notes.keyword_collection_link":
            self._materialize_collection_link(cursor, operation, payload)
            return None
        return self._materialize_folder_link(cursor, dataset_id, operation, payload)

    def _materialize_keyword(
        self,
        cursor: sqlite3.Cursor,
        dataset_id: str,
        sync_id: str,
        operation: str,
        payload: Mapping[str, object],
    ) -> int | None:
        row = self.get_resource_by_sync_id("notes.keyword", sync_id, cursor=cursor)
        if operation == "tombstone":
            if row is None:
                raise NotesOrganizationRepositoryError(
                    "missing_dependency", "tombstoned keyword is unknown locally"
                )
            cursor.execute(
                "UPDATE keywords SET deleted = 1, version = version + 1, last_modified = ? WHERE id = ?",
                (_utc_timestamp(), row["id"]),
            )
            return int(row["id"])
        name = str(payload["keyword"])
        portable_collision_key(name, maximum=100)
        self._reject_resource_name_collision(
            cursor,
            dataset_id=dataset_id,
            domain="notes.keyword",
            table="keywords",
            column="keyword",
            name=name,
            sync_id=sync_id,
            maximum=100,
        )
        now = _utc_timestamp()
        if row is None:
            cursor.execute(
                "INSERT INTO keywords(keyword, created_at, last_modified, deleted, client_id, version, sync_id) VALUES (?, ?, ?, 0, ?, 1, ?)",
                (name, now, now, self.db.client_id, sync_id),
            )
            if cursor.lastrowid is None:  # pragma: no cover - SQLite contract
                raise NotesOrganizationRepositoryError(
                    "projection_failed", "inserted keyword has no local identity"
                )
            return int(cursor.lastrowid)
        cursor.execute(
            "UPDATE keywords SET keyword = ?, deleted = 0, version = version + 1, last_modified = ? WHERE id = ?",
            (name, now, row["id"]),
        )
        return int(row["id"])

    def _materialize_collection(
        self,
        cursor: sqlite3.Cursor,
        dataset_id: str,
        sync_id: str,
        operation: str,
        payload: Mapping[str, object],
    ) -> int | None:
        row = self.get_resource_by_sync_id(
            "notes.keyword_collection", sync_id, cursor=cursor
        )
        if operation == "tombstone":
            if row is None:
                raise NotesOrganizationRepositoryError(
                    "missing_dependency", "tombstoned collection is unknown locally"
                )
            cursor.execute(
                "UPDATE keyword_collections SET deleted = 1, version = version + 1, last_modified = ? WHERE id = ?",
                (_utc_timestamp(), row["id"]),
            )
            return int(row["id"])
        name = str(payload["name"])
        portable_collision_key(name, maximum=255)
        parent = self._parent_resource(
            cursor, "notes.keyword_collection", payload.get("parent_sync_id")
        )
        if (
            row is not None
            and parent is not None
            and self._collection_descends_from(
                cursor, int(parent["id"]), int(row["id"])
            )
        ):
            raise NotesOrganizationRepositoryError(
                "hierarchy_cycle", "collection hierarchy contains a cycle"
            )
        portable_relative_path(self._collection_segments(cursor, parent) + (name,))
        self._reject_resource_name_collision(
            cursor,
            dataset_id=dataset_id,
            domain="notes.keyword_collection",
            table="keyword_collections",
            column="name",
            name=name,
            sync_id=sync_id,
            maximum=255,
        )
        now = _utc_timestamp()
        parent_id = int(parent["id"]) if parent is not None else None
        if row is None:
            cursor.execute(
                "INSERT INTO keyword_collections(name, parent_id, created_at, last_modified, deleted, client_id, version, sync_id) VALUES (?, ?, ?, ?, 0, ?, 1, ?)",
                (name, parent_id, now, now, self.db.client_id, sync_id),
            )
            if cursor.lastrowid is None:  # pragma: no cover - SQLite contract
                raise NotesOrganizationRepositoryError(
                    "projection_failed", "inserted collection has no local identity"
                )
            return int(cursor.lastrowid)
        cursor.execute(
            "UPDATE keyword_collections SET name = ?, parent_id = ?, deleted = 0, version = version + 1, last_modified = ? WHERE id = ?",
            (name, parent_id, now, row["id"]),
        )
        return int(row["id"])

    def _materialize_folder(
        self,
        cursor: sqlite3.Cursor,
        dataset_id: str,
        sync_id: str,
        operation: str,
        payload: Mapping[str, object],
    ) -> str | None:
        row = self.get_resource_by_sync_id("notes.folder", sync_id, cursor=cursor)
        if operation == "tombstone":
            if row is None:
                raise NotesOrganizationRepositoryError(
                    "missing_dependency", "tombstoned folder is unknown locally"
                )
            cursor.execute(
                "UPDATE note_folders SET deleted = 1, version = version + 1, "
                "modified_at = ? WHERE id = ?",
                (_utc_timestamp(), row["id"]),
            )
            return str(row["id"])
        name = str(payload["name"])
        portable_collision_key(name, maximum=500)
        parent = self._parent_resource(
            cursor, "notes.folder", payload.get("parent_sync_id")
        )
        if (
            row is not None
            and parent is not None
            and self._folder_descends_from(cursor, str(parent["id"]), str(row["id"]))
        ):
            raise NotesOrganizationRepositoryError(
                "hierarchy_cycle", "folder hierarchy contains a cycle"
            )
        segments = self._folder_segments(cursor, parent) + (name,)
        portable_relative_path(segments)
        parent_path = str(parent["path"]) if parent is not None else ""
        parent_normalized = str(parent["normalized_path"]) if parent is not None else ""
        display = name.strip()
        local_key = unicodedata.normalize("NFKC", display).casefold()
        path = f"{parent_path}/{display}"
        normalized_path = f"{parent_normalized}/{local_key}"
        now = _utc_timestamp()
        parent_id = str(parent["id"]) if parent is not None else None
        if row is None:
            collision = cursor.execute(
                "SELECT id FROM note_folders WHERE normalized_path = ? AND deleted = 0",
                (normalized_path,),
            ).fetchone()
            if collision is not None:
                self._record_adoption_review(
                    cursor,
                    dataset_id,
                    domain="notes.folder",
                    local_object_id=str(collision["id"]),
                    remote_object_id=sync_id,
                    collision_key=portable_relative_path(segments),
                    portable_path=portable_relative_path(segments),
                    display=display,
                )
                raise NotesOrganizationRepositoryError(
                    "local_representation_collision",
                    "portable folder cannot be represented without merging distinct resources",
                )
            local_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO note_folders(id, parent_id, name, normalized_name, path, normalized_path, version, deleted, created_at, modified_at, sync_id) VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, ?, ?)",
                (
                    local_id,
                    parent_id,
                    display,
                    local_key,
                    path,
                    normalized_path,
                    now,
                    now,
                    sync_id,
                ),
            )
            return local_id
        rewritten = self._preflight_folder_rewrite(
            cursor,
            dataset_id=dataset_id,
            remote_object_id=sync_id,
            target=row,
            target_parent_id=parent_id,
            target_name=display,
            target_path=path,
            target_normalized_path=normalized_path,
            target_portable_segments=segments,
        )
        for rewritten_row in rewritten:
            is_target = str(rewritten_row["id"]) == str(row["id"])
            if not is_target and not bool(rewritten_row["changed"]):
                continue
            cursor.execute(
                "UPDATE note_folders SET parent_id = ?, name = ?, normalized_name = ?, "
                "path = ?, normalized_path = ?, deleted = ?, "
                f"version = version + {1 if is_target else 0}, modified_at = ? "
                "WHERE id = ?",
                (
                    rewritten_row["parent_id"],
                    rewritten_row["name"],
                    rewritten_row["normalized_name"],
                    rewritten_row["path"],
                    rewritten_row["normalized_path"],
                    rewritten_row["deleted"],
                    now,
                    rewritten_row["id"],
                ),
            )
        return str(row["id"])

    def _materialize_keyword_link(
        self, cursor: sqlite3.Cursor, operation: str, payload: Mapping[str, object]
    ) -> None:
        subject_type, subject_id = (
            str(payload["subject_type"]),
            str(payload["subject_id"]),
        )
        table = "note_keywords" if subject_type == "note" else "conversation_keywords"
        column = "note_id" if subject_type == "note" else "conversation_id"
        keyword = self.get_resource_by_sync_id(
            "notes.keyword", str(payload["keyword_sync_id"]), cursor=cursor
        )
        if operation == "tombstone":
            if keyword is None:
                raise NotesOrganizationRepositoryError(
                    "missing_dependency", "tombstoned keyword link is unknown locally"
                )
            self._require_subject_exists(
                cursor,
                "notes" if subject_type == "note" else "conversations",
                subject_id,
            )
            cursor.execute(
                f"DELETE FROM {table} WHERE {column} = ? AND keyword_id = ?",
                (subject_id, keyword["id"]),
            )
            return None
        if keyword is None or bool(keyword["deleted"]):
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "referenced resource is missing or inactive"
            )
        self._require_subject(
            cursor, "notes" if subject_type == "note" else "conversations", subject_id
        )
        cursor.execute(
            f"INSERT OR IGNORE INTO {table}({column}, keyword_id, created_at) VALUES (?, ?, ?)",
            (subject_id, keyword["id"], _utc_timestamp()),
        )
        return None

    def _materialize_collection_link(
        self, cursor: sqlite3.Cursor, operation: str, payload: Mapping[str, object]
    ) -> None:
        collection = self.get_resource_by_sync_id(
            "notes.keyword_collection",
            str(payload["collection_sync_id"]),
            cursor=cursor,
        )
        keyword = self.get_resource_by_sync_id(
            "notes.keyword", str(payload["keyword_sync_id"]), cursor=cursor
        )
        if operation == "tombstone":
            if collection is None or keyword is None:
                raise NotesOrganizationRepositoryError(
                    "missing_dependency",
                    "tombstoned collection link has a missing endpoint",
                )
            cursor.execute(
                "DELETE FROM collection_keywords WHERE collection_id = ? AND keyword_id = ?",
                (collection["id"], keyword["id"]),
            )
            return None
        if (
            collection is None
            or bool(collection["deleted"])
            or keyword is None
            or bool(keyword["deleted"])
        ):
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "referenced resource is missing or inactive"
            )
        cursor.execute(
            "INSERT OR IGNORE INTO collection_keywords(collection_id, keyword_id, created_at) VALUES (?, ?, ?)",
            (collection["id"], keyword["id"], _utc_timestamp()),
        )
        return None

    def _materialize_folder_link(
        self,
        cursor: sqlite3.Cursor,
        dataset_id: str,
        operation: str,
        payload: Mapping[str, object],
    ) -> str | None:
        note_id, folder_sync_id = (
            str(payload["note_id"]),
            str(payload["folder_sync_id"]),
        )
        folder = self.get_resource_by_sync_id(
            "notes.folder", folder_sync_id, cursor=cursor
        )
        if operation == "tombstone":
            note = cursor.execute(
                "SELECT id FROM notes WHERE id = ?", (note_id,)
            ).fetchone()
            if note is None or folder is None:
                raise NotesOrganizationRepositoryError(
                    "missing_dependency",
                    "tombstoned folder link has a missing endpoint",
                )
            now = _utc_timestamp()
            cursor.execute(
                "UPDATE note_folder_memberships SET deleted = 1, "
                "version = version + 1, modified_at = ? WHERE folder_id = ? "
                "AND note_id = ? AND ownership = 'manual' AND owner_id = '' "
                "AND deleted = 0",
                (now, folder["id"], note_id),
            )
            cursor.execute(
                "INSERT OR IGNORE INTO note_folder_sync_suppressions(note_id, folder_sync_id, created_at) VALUES (?, ?, ?)",
                (note_id, folder_sync_id, now),
            )
            return None
        self._require_subject(cursor, "notes", note_id)
        if folder is None or bool(folder["deleted"]):
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "referenced resource is missing or inactive"
            )
        existing = cursor.execute(
            "SELECT id, deleted FROM note_folder_memberships WHERE folder_id = ? "
            "AND note_id = ? AND ownership = 'manual' AND owner_id = '' "
            "ORDER BY deleted, id LIMIT 1",
            (folder["id"], note_id),
        ).fetchone()
        membership_id = (
            self._new_membership_id(cursor) if existing is None else str(existing["id"])
        )
        cursor.execute(
            "DELETE FROM note_folder_sync_suppressions WHERE note_id = ? AND folder_sync_id = ?",
            (note_id, folder_sync_id),
        )
        now = _utc_timestamp()
        if existing is None:
            cursor.execute(
                "INSERT INTO note_folder_memberships(id, folder_id, note_id, ownership, "
                "owner_id, owner_active, version, deleted, created_at, modified_at) "
                "VALUES (?, ?, ?, 'manual', '', 1, 1, 0, ?, ?)",
                (membership_id, folder["id"], note_id, now, now),
            )
            return membership_id
        if bool(existing["deleted"]):
            cursor.execute(
                "UPDATE note_folder_memberships SET deleted = 0, version = version + 1, "
                "modified_at = ? WHERE id = ?",
                (now, existing["id"]),
            )
        return str(existing["id"])

    def _write_head(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        domain: str,
        object_id: str,
        operation: str,
        payload_json: str,
        payload_hash: str,
        object_revision: int,
        object_hash: str,
        server_cursor: str,
        apply_state: str,
        local_id: str | int | None = None,
        reason_code: str | None = None,
    ) -> ApplyResult:
        now = _utc_timestamp()
        applied_at = now if apply_state == "applied" else None
        cursor.execute(
            """
            INSERT INTO notes_organization_heads(server_profile_id, dataset_id, domain, object_id, operation, schema_version, encryption_policy, payload_json, payload_hash, object_revision, object_hash, server_cursor, deleted, apply_state, applied_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, 'server_trusted_v1', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(server_profile_id, dataset_id, domain, object_id) DO UPDATE SET
              operation=excluded.operation, payload_json=excluded.payload_json,
              payload_hash=excluded.payload_hash, object_revision=excluded.object_revision,
              object_hash=excluded.object_hash, server_cursor=excluded.server_cursor,
              deleted=excluded.deleted, apply_state=excluded.apply_state,
              applied_at=excluded.applied_at, updated_at=excluded.updated_at
        """,
            (
                self.server_profile_id,
                dataset_id,
                domain,
                object_id,
                operation,
                payload_json,
                payload_hash,
                object_revision,
                object_hash,
                server_cursor,
                int(operation == "tombstone"),
                apply_state,
                applied_at,
                now,
            ),
        )
        return ApplyResult(
            "applied" if apply_state == "applied" else "blocked", local_id, reason_code
        )

    def _parent_resource(
        self, cursor: sqlite3.Cursor, domain: str, parent_sync_id: object
    ) -> sqlite3.Row | None:
        if parent_sync_id is None:
            return None
        row = self.get_resource_by_sync_id(domain, str(parent_sync_id), cursor=cursor)
        if row is None or bool(row["deleted"]):
            raise NotesOrganizationRepositoryError(
                "missing_parent", "parent resource is missing or inactive"
            )
        return row

    def _required_active_resource(
        self, cursor: sqlite3.Cursor, domain: str, sync_id: str
    ) -> sqlite3.Row:
        row = self.get_resource_by_sync_id(domain, sync_id, cursor=cursor)
        if row is None or bool(row["deleted"]):
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "referenced resource is missing or inactive"
            )
        return row

    @staticmethod
    def _require_subject(cursor: sqlite3.Cursor, table: str, subject_id: str) -> None:
        row = cursor.execute(
            f"SELECT id FROM {table} WHERE id = ? AND deleted = 0", (subject_id,)
        ).fetchone()
        if row is None:
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "referenced subject is missing or inactive"
            )

    @staticmethod
    def _require_subject_exists(
        cursor: sqlite3.Cursor, table: str, subject_id: str
    ) -> None:
        row = cursor.execute(
            f"SELECT id FROM {table} WHERE id = ?", (subject_id,)
        ).fetchone()
        if row is None:
            raise NotesOrganizationRepositoryError(
                "missing_dependency", "referenced subject is unknown locally"
            )

    def _reject_resource_name_collision(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        domain: str,
        table: str,
        column: str,
        name: str,
        sync_id: str,
        maximum: int,
    ) -> None:
        key = portable_collision_key(name, maximum=maximum)
        collision = cursor.execute(
            f"SELECT id, {column}, sync_id FROM {table} "
            f"WHERE {column} = ? COLLATE NOCASE "
            "AND (sync_id IS NULL OR sync_id <> ?) ORDER BY id LIMIT 1",
            (name, sync_id),
        ).fetchone()
        rows = cursor.execute(
            f"SELECT id, {column}, sync_id FROM {table} WHERE deleted = 0 ORDER BY id"
        ).fetchall()
        if collision is None:
            collision = next(
                (
                    row
                    for row in rows
                    if portable_collision_key(str(row[column]), maximum=maximum) == key
                    and row["sync_id"] != sync_id
                ),
                None,
            )
        if collision is not None:
            self._record_adoption_review(
                cursor,
                dataset_id,
                domain=domain,
                local_object_id=str(collision["id"]),
                remote_object_id=sync_id,
                collision_key=key,
                display=name.strip(),
                portable_path=None,
            )
            raise NotesOrganizationRepositoryError(
                "local_representation_collision",
                "portable resource collides with existing local organization",
            )

    def _require_owned_cursor(self, cursor: sqlite3.Cursor) -> None:
        if not isinstance(cursor, sqlite3.Cursor):
            raise TypeError("cursor must be a sqlite3.Cursor")
        if cursor.connection is not self.db.get_connection():
            raise ValueError("cursor must belong to the repository-owned connection")

    @staticmethod
    def _new_membership_id(cursor: sqlite3.Cursor) -> str:
        for _ in range(LOCAL_ID_ALLOCATION_ATTEMPTS):
            candidate = str(uuid.uuid4())
            if (
                cursor.execute(
                    "SELECT 1 FROM note_folder_memberships WHERE id = ?", (candidate,)
                ).fetchone()
                is None
            ):
                return candidate
        raise NotesOrganizationRepositoryError(
            "projection_id_exhausted",
            "could not allocate a unique local membership identity",
        )

    @staticmethod
    def _collection_descends_from(
        cursor: sqlite3.Cursor, child_id: int, ancestor_id: int
    ) -> bool:
        row = cursor.execute(
            "WITH RECURSIVE ancestors(id) AS (SELECT ? UNION ALL SELECT parent_id FROM keyword_collections JOIN ancestors ON keyword_collections.id = ancestors.id WHERE parent_id IS NOT NULL) SELECT 1 FROM ancestors WHERE id = ? LIMIT 1",
            (child_id, ancestor_id),
        ).fetchone()
        return row is not None

    @staticmethod
    def _folder_descends_from(
        cursor: sqlite3.Cursor, child_id: str, ancestor_id: str
    ) -> bool:
        row = cursor.execute(
            "WITH RECURSIVE ancestors(id) AS (SELECT ? UNION ALL SELECT parent_id FROM note_folders JOIN ancestors ON note_folders.id = ancestors.id WHERE parent_id IS NOT NULL) SELECT 1 FROM ancestors WHERE id = ? LIMIT 1",
            (child_id, ancestor_id),
        ).fetchone()
        return row is not None

    @staticmethod
    def _folder_segments(
        cursor: sqlite3.Cursor, parent: sqlite3.Row | None
    ) -> tuple[str, ...]:
        if parent is None:
            return ()
        rows = cursor.execute(
            "WITH RECURSIVE lineage(id, parent_id, name, depth) AS (SELECT id, parent_id, name, 0 FROM note_folders WHERE id = ? UNION ALL SELECT f.id, f.parent_id, f.name, lineage.depth + 1 FROM note_folders f JOIN lineage ON lineage.parent_id = f.id) SELECT name FROM lineage ORDER BY depth DESC",
            (parent["id"],),
        ).fetchall()
        return tuple(str(row["name"]) for row in rows)

    @staticmethod
    def _collection_segments(
        cursor: sqlite3.Cursor, parent: sqlite3.Row | None
    ) -> tuple[str, ...]:
        if parent is None:
            return ()
        rows = cursor.execute(
            "WITH RECURSIVE lineage(id, parent_id, name, depth) AS ("
            "SELECT id, parent_id, name, 0 FROM keyword_collections WHERE id = ? "
            "UNION ALL SELECT c.id, c.parent_id, c.name, lineage.depth + 1 "
            "FROM keyword_collections c JOIN lineage ON lineage.parent_id = c.id) "
            "SELECT name FROM lineage ORDER BY depth DESC",
            (parent["id"],),
        ).fetchall()
        return tuple(str(row["name"]) for row in rows)

    def _preflight_folder_rewrite(
        self,
        cursor: sqlite3.Cursor,
        *,
        dataset_id: str,
        remote_object_id: str,
        target: sqlite3.Row,
        target_parent_id: str | None,
        target_name: str,
        target_path: str,
        target_normalized_path: str,
        target_portable_segments: tuple[str, ...],
    ) -> tuple[dict[str, object], ...]:
        rows = cursor.execute(
            "WITH RECURSIVE subtree(id, parent_id, name, path, normalized_path, deleted, depth) AS ("
            "SELECT id, parent_id, name, path, normalized_path, deleted, 0 "
            "FROM note_folders WHERE id = ? UNION ALL "
            "SELECT f.id, f.parent_id, f.name, f.path, f.normalized_path, f.deleted, "
            "subtree.depth + 1 "
            "FROM note_folders f JOIN subtree ON f.parent_id = subtree.id) "
            "SELECT * FROM subtree ORDER BY depth, id",
            (target["id"],),
        ).fetchall()
        rewritten: dict[str, dict[str, object]] = {}
        portable_segments: dict[str, tuple[str, ...]] = {}
        target_id = str(target["id"])
        for row in rows:
            row_id = str(row["id"])
            if row_id == target_id:
                name = target_name
                parent_id = target_parent_id
                path = target_path
                normalized_path = target_normalized_path
                segments = target_portable_segments
                deleted = 0
            else:
                parent_id = str(row["parent_id"])
                parent = rewritten[parent_id]
                name = str(row["name"])
                normalized_name = unicodedata.normalize("NFKC", name).casefold()
                path = f"{parent['path']}/{name}"
                normalized_path = f"{parent['normalized_path']}/{normalized_name}"
                segments = portable_segments[parent_id] + (name,)
                deleted = int(row["deleted"])
            portable_relative_path(segments)
            rewritten[row_id] = {
                "id": row_id,
                "parent_id": parent_id,
                "name": name,
                "normalized_name": unicodedata.normalize("NFKC", name).casefold(),
                "path": path,
                "normalized_path": normalized_path,
                "deleted": deleted,
                "portable_path": portable_relative_path(segments),
                "changed": row_id == target_id
                or str(row["path"]) != path
                or str(row["normalized_path"]) != normalized_path,
            }
            portable_segments[row_id] = segments

        active_by_path: dict[str, dict[str, object]] = {}
        for item in rewritten.values():
            if bool(item["deleted"]):
                continue
            normalized_path = str(item["normalized_path"])
            collision = active_by_path.get(normalized_path)
            if collision is not None:
                portable_path = str(item["portable_path"])
                self._record_adoption_review(
                    cursor,
                    dataset_id,
                    domain="notes.folder",
                    local_object_id=str(collision["id"]),
                    remote_object_id=remote_object_id,
                    collision_key=portable_path,
                    portable_path=portable_path,
                    display=str(item["name"]),
                )
                raise NotesOrganizationRepositoryError(
                    "local_representation_collision",
                    "rewritten subtree contains colliding local paths",
                )
            active_by_path[normalized_path] = item
        subtree_ids = tuple(rewritten)
        placeholders = ",".join("?" for _ in subtree_ids)
        external = cursor.execute(
            "SELECT id, normalized_path FROM note_folders WHERE deleted = 0 "
            f"AND id NOT IN ({placeholders}) ORDER BY id",
            subtree_ids,
        ).fetchall()
        external_collision = next(
            (
                (row, active_by_path[str(row["normalized_path"])])
                for row in external
                if str(row["normalized_path"]) in active_by_path
            ),
            None,
        )
        if external_collision is not None:
            external_row, rewritten_row = external_collision
            portable_path = str(rewritten_row["portable_path"])
            self._record_adoption_review(
                cursor,
                dataset_id,
                domain="notes.folder",
                local_object_id=str(external_row["id"]),
                remote_object_id=remote_object_id,
                collision_key=portable_path,
                portable_path=portable_path,
                display=str(rewritten_row["name"]),
            )
            raise NotesOrganizationRepositoryError(
                "local_representation_collision",
                "rewritten subtree collides with an active local folder",
            )
        return tuple(rewritten.values())

    def _record_adoption_review(
        self,
        cursor: sqlite3.Cursor,
        dataset_id: str,
        *,
        domain: str,
        local_object_id: str,
        remote_object_id: str | None,
        collision_key: str,
        portable_path: str | None,
        display: str,
    ) -> None:
        now = _utc_timestamp()
        review_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self.server_profile_id}:{dataset_id}:{domain}:{local_object_id}",
            )
        )
        cursor.execute(
            "INSERT OR IGNORE INTO notes_organization_adoption_reviews(review_id, server_profile_id, dataset_id, domain, local_object_id, remote_object_id, collision_key, display_name, portable_path, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)",
            (
                review_id,
                self.server_profile_id,
                dataset_id,
                domain,
                local_object_id,
                remote_object_id,
                collision_key,
                display,
                portable_path,
                now,
                now,
            ),
        )


def _dependency_refs(
    domain: str, payload: Mapping[str, object]
) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    if domain == "notes.keyword_link":
        refs.append(
            {"domain": "notes.keyword", "object_id": str(payload["keyword_sync_id"])}
        )
        refs.append(
            {
                "domain": (
                    "notes.note"
                    if payload["subject_type"] == "note"
                    else "chat.conversation"
                ),
                "object_id": str(payload["subject_id"]),
            }
        )
    elif (
        domain in {"notes.keyword_collection", "notes.folder"}
        and payload.get("parent_sync_id") is not None
    ):
        refs.append({"domain": domain, "object_id": str(payload["parent_sync_id"])})
    elif domain == "notes.keyword_collection_link":
        refs.extend(
            (
                {
                    "domain": "notes.keyword_collection",
                    "object_id": str(payload["collection_sync_id"]),
                },
                {
                    "domain": "notes.keyword",
                    "object_id": str(payload["keyword_sync_id"]),
                },
            )
        )
    elif domain == "notes.folder_link":
        refs.extend(
            (
                {"domain": "notes.note", "object_id": str(payload["note_id"])},
                {"domain": "notes.folder", "object_id": str(payload["folder_sync_id"])},
            )
        )
    return refs


def _json_scalar_values(value: object) -> set[str]:
    """Return stringified scalar values from a small organization payload."""

    if isinstance(value, Mapping):
        result: set[str] = set()
        for child in value.values():
            result.update(_json_scalar_values(child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = set()
        for child in value:
            result.update(_json_scalar_values(child))
        return result
    return {str(value)} if value is not None else set()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = [
    "ApplyResult",
    "NotesOrganizationRepository",
    "NotesOrganizationRepositoryError",
    "portable_collision_key",
    "portable_relative_path",
]
