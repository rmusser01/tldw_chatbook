"""Bounded, resumable inventory of adopted legacy Notes organization state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
    NotesOrganizationRepositoryError,
)
from tldw_chatbook.Sync_Interop.notes_organization import (
    organization_link_id,
    validate_organization_object_id,
    validate_resource_sync_id,
)

_PHASES = ("resources", "links", "tombstones")
_RESOURCE_TABLES = {
    "notes.keyword": ("keywords", "keyword"),
    "notes.keyword_collection": ("keyword_collections", "name"),
    "notes.folder": ("note_folders", "name"),
}


@dataclass(frozen=True)
class InventoryRunResult:
    """Outcome of one inventory run."""

    status: Literal["held", "complete"]
    reason_code: str | None = None
    skipped_dependencies: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class _Intent:
    phase: str
    domain: str
    object_id: str
    operation: str
    payload: Mapping[str, object]
    source_version: int
    merge: tuple[str, str] | None = None

    @property
    def key(self) -> str:
        return f"{self.domain}|{self.operation}|{self.object_id}"


@dataclass(frozen=True)
class _Snapshot:
    digest: str
    phases: Mapping[str, tuple[_Intent, ...]]
    skipped_dependencies: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class _ResourceScan:
    final_ids: Mapping[tuple[str, str], str]
    merges: Mapping[tuple[str, str], str]
    rows: Mapping[str, tuple[sqlite3.Row, ...]]
    deleted: frozenset[tuple[str, str]]
    sync_aliases: Mapping[tuple[str, str], str]


class LegacyNotesOrganizationInventory:
    """Create immutable organization intents one source-stable object at a time."""

    def __init__(
        self,
        repository: NotesOrganizationRepository,
        *,
        dataset_id: str,
        enrolled_note_ids: Iterable[str],
        enrolled_conversation_ids: Iterable[str],
    ) -> None:
        if not isinstance(repository, NotesOrganizationRepository):
            raise TypeError("repository must be a NotesOrganizationRepository")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError("dataset_id must be non-blank text")
        self.repository = repository
        self.dataset_id = dataset_id.strip()
        self.enrolled_note_ids = frozenset(str(value) for value in enrolled_note_ids)
        self.enrolled_conversation_ids = frozenset(
            str(value) for value in enrolled_conversation_ids
        )

    def run(
        self,
        *,
        after_commit: Callable[[str, str | None], None] | None = None,
    ) -> InventoryRunResult:
        """Resume inventory, invoking the optional failure hook after each commit."""

        callback = after_commit or (lambda _phase, _key: None)
        skipped: tuple[tuple[str, str, str], ...] = ()
        while True:
            committed: tuple[str, str | None] | None = None
            outcome: InventoryRunResult | None = None
            with self.repository.db.transaction() as cursor:
                checkpoint = self._checkpoint(cursor)
                prerequisite = self._prerequisite_reason(cursor, checkpoint)
                if prerequisite is not None:
                    outcome = InventoryRunResult("held", prerequisite)
                elif checkpoint is None:  # pragma: no cover - prerequisite guards this
                    outcome = InventoryRunResult("held", "bootstrap_pull_incomplete")
                elif checkpoint["inventory_phase"] == "complete":
                    snapshot = self._snapshot(cursor)
                    baseline, final_key = self._parse_token(
                        checkpoint["last_inventory_key"]
                    )
                    if baseline != snapshot.digest:
                        outcome = InventoryRunResult(
                            "held",
                            "inventory_source_changed",
                            snapshot.skipped_dependencies,
                        )
                    elif final_key is not None:
                        raise NotesOrganizationRepositoryError(
                            "invalid_inventory_checkpoint",
                            "completed inventory checkpoint retains an object key",
                        )
                    else:
                        outcome = InventoryRunResult(
                            "complete",
                            skipped_dependencies=snapshot.skipped_dependencies,
                        )
                else:
                    # ponytail: re-hash the source for each bounded commit; add a
                    # snapshot table only if large inventories make this measurable.
                    snapshot = self._snapshot(cursor)
                    skipped = snapshot.skipped_dependencies
                    phase = str(checkpoint["inventory_phase"])
                    raw_token: str | None = checkpoint["last_inventory_key"]
                    if phase == "not_started":
                        token = self._token(snapshot.digest, None)
                        self.repository.advance_inventory_checkpoint(
                            cursor,
                            dataset_id=self.dataset_id,
                            expected_phase="not_started",
                            expected_key=None,
                            inventory_phase="resources",
                            last_inventory_key=token,
                        )
                        committed = ("resources", None)
                    else:
                        baseline, last_key = self._parse_token(raw_token)
                        if baseline != snapshot.digest:
                            outcome = InventoryRunResult(
                                "held",
                                "inventory_source_changed",
                                snapshot.skipped_dependencies,
                            )
                        else:
                            entries = snapshot.phases[phase]
                            next_entry = self._next_entry(entries, last_key)
                            if next_entry is not None:
                                if next_entry.merge is not None:
                                    local_id, remote_id = next_entry.merge
                                    self.repository.apply_resolved_inventory_merge(
                                        cursor,
                                        dataset_id=self.dataset_id,
                                        domain=next_entry.domain,
                                        local_object_id=local_id,
                                        remote_object_id=remote_id,
                                    )
                                if not self._matches_applied_remote_head(
                                    cursor, next_entry
                                ):
                                    self.repository.record_intent(
                                        cursor,
                                        profile=self.repository.server_profile_id,
                                        dataset=self.dataset_id,
                                        domain=next_entry.domain,
                                        object_id=next_entry.object_id,
                                        operation=next_entry.operation,
                                        payload=next_entry.payload,
                                        source_version=next_entry.source_version,
                                    )
                                next_token: str | None = self._token(
                                    snapshot.digest, next_entry.key
                                )
                                self.repository.advance_inventory_checkpoint(
                                    cursor,
                                    dataset_id=self.dataset_id,
                                    expected_phase=phase,
                                    expected_key=raw_token,
                                    inventory_phase=phase,
                                    last_inventory_key=next_token,
                                )
                                committed = (phase, next_entry.key)
                            else:
                                next_phase = self._next_phase(phase)
                                next_token = self._token(snapshot.digest, None)
                                self.repository.advance_inventory_checkpoint(
                                    cursor,
                                    dataset_id=self.dataset_id,
                                    expected_phase=phase,
                                    expected_key=raw_token,
                                    inventory_phase=next_phase,
                                    last_inventory_key=next_token,
                                )
                                committed = (next_phase, None)
            if outcome is not None:
                return outcome
            if committed is None:  # pragma: no cover - every branch above terminates
                raise RuntimeError("inventory made no progress")
            callback(*committed)
            if committed[0] == "complete":
                return InventoryRunResult("complete", skipped_dependencies=skipped)

    def _matches_applied_remote_head(
        self, cursor: sqlite3.Cursor, intent: _Intent
    ) -> bool:
        """Return whether inventory merely reflects state already pulled.

        Inventory publishes pre-enrollment local organization state. A resource
        or link materialized by the bootstrap pull is already authoritative on
        the server; re-emitting it would reuse the deterministic intent ID with
        device-specific base metadata and trigger a server idempotency conflict.
        """

        head = cursor.execute(
            "SELECT operation, payload_json FROM notes_organization_heads "
            "WHERE server_profile_id = ? AND dataset_id = ? AND domain = ? "
            "AND object_id = ? AND apply_state = 'applied'",
            (
                self.repository.server_profile_id,
                self.dataset_id,
                intent.domain,
                intent.object_id,
            ),
        ).fetchone()
        return (
            head is not None
            and str(head["operation"]) == intent.operation
            and str(head["payload_json"]) == _canonical_json(intent.payload)
        )

    def _checkpoint(self, cursor: sqlite3.Cursor) -> sqlite3.Row | None:
        return cursor.execute(
            """
            SELECT * FROM notes_organization_sync_checkpoints
             WHERE server_profile_id = ? AND dataset_id = ?
            """,
            (self.repository.server_profile_id, self.dataset_id),
        ).fetchone()

    def _prerequisite_reason(
        self, cursor: sqlite3.Cursor, checkpoint: sqlite3.Row | None
    ) -> str | None:
        if checkpoint is None:
            return "bootstrap_pull_incomplete"
        phase = str(checkpoint["inventory_phase"])
        if phase == "complete":
            return None
        if (
            checkpoint["local_state"] != "adoption_review"
            or checkpoint["server_state"] != "ready"
            or not checkpoint["bootstrap_id"]
            or checkpoint["error_code"] is not None
            or int(checkpoint["captured_count"]) != int(checkpoint["expected_count"])
        ):
            return "bootstrap_pull_incomplete"
        open_review = cursor.execute(
            """
            SELECT 1 FROM notes_organization_adoption_reviews
             WHERE server_profile_id = ? AND dataset_id = ? AND state = 'open'
             LIMIT 1
            """,
            (self.repository.server_profile_id, self.dataset_id),
        ).fetchone()
        return "adoption_review_required" if open_review is not None else None

    def _snapshot(self, cursor: sqlite3.Cursor) -> _Snapshot:
        resource_scan = self._scan_resources(cursor)
        resources = self._resource_intents(resource_scan)
        links, link_tombstones, skipped, evidence = self._link_intents(
            cursor, resource_scan.final_ids, resource_scan.sync_aliases
        )
        resource_tombstones = tuple(
            _Intent(
                "tombstones",
                intent.domain,
                intent.object_id,
                "tombstone",
                {},
                intent.source_version,
            )
            for intent in resources
            if (intent.domain, intent.object_id) in resource_scan.deleted
        )
        tombstones = tuple(
            sorted(
                (*link_tombstones, *resource_tombstones),
                key=lambda item: (item.domain in _RESOURCE_TABLES, item.key),
            )
        )
        phases = {
            "resources": resources,
            "links": links,
            "tombstones": tombstones,
        }
        material = {
            "dependencies": {
                "notes.note": sorted(self.enrolled_note_ids),
                "chat.conversation": sorted(self.enrolled_conversation_ids),
            },
            "evidence": evidence,
            "intents": [
                {
                    "phase": item.phase,
                    "domain": item.domain,
                    "object_id": item.object_id,
                    "operation": item.operation,
                    "payload": item.payload,
                    "source_version": item.source_version,
                    "merge": item.merge,
                }
                for phase in _PHASES
                for item in phases[phase]
            ],
            "skipped": skipped,
        }
        digest = hashlib.sha256(_canonical_json(material).encode()).hexdigest()
        return _Snapshot(digest, phases, skipped)

    def _scan_resources(self, cursor: sqlite3.Cursor) -> _ResourceScan:
        reviews = cursor.execute(
            """
            SELECT domain, local_object_id, remote_object_id, resolution
              FROM notes_organization_adoption_reviews
             WHERE server_profile_id = ? AND dataset_id = ? AND state = 'resolved'
             ORDER BY domain, local_object_id
            """,
            (self.repository.server_profile_id, self.dataset_id),
        ).fetchall()
        resolved = {
            (str(row["domain"]), str(row["local_object_id"])): row for row in reviews
        }
        final_ids: dict[tuple[str, str], str] = {}
        merges: dict[tuple[str, str], str] = {}
        rows_by_domain: dict[str, tuple[sqlite3.Row, ...]] = {}
        deleted: set[tuple[str, str]] = set()
        sync_aliases: dict[tuple[str, str], str] = {}
        for domain, (table, name_column) in _RESOURCE_TABLES.items():
            parent_column = "NULL AS parent_id" if domain == "notes.keyword" else "parent_id"
            rows = cursor.execute(
                f"SELECT id, {parent_column}, {name_column} AS name, version, "
                f"deleted, sync_id FROM {table} ORDER BY id"
            ).fetchall()
            ordered_rows = _parent_first(rows)
            rows_by_domain[domain] = ordered_rows
            for row in ordered_rows:
                local_id = str(row["id"])
                review = resolved.get((domain, local_id))
                if review is not None and review["resolution"] == "keep_local":
                    continue
                if row["parent_id"] is not None and (
                    domain,
                    str(row["parent_id"]),
                ) not in final_ids:
                    continue
                original_sync_id = (
                    str(row["sync_id"]) if row["sync_id"] is not None else ""
                )
                sync_id = original_sync_id
                if review is not None and review["resolution"] == "merge":
                    sync_id = str(review["remote_object_id"])
                    merges[(domain, local_id)] = sync_id
                validate_resource_sync_id(sync_id)
                final_ids[(domain, local_id)] = sync_id
                sync_aliases[(domain, original_sync_id)] = sync_id
                if bool(row["deleted"]):
                    deleted.add((domain, sync_id))
        for domain in _RESOURCE_TABLES:
            domain_ids = [
                sync_id
                for (candidate_domain, _local_id), sync_id in final_ids.items()
                if candidate_domain == domain
            ]
            if len(domain_ids) != len(set(domain_ids)):
                raise NotesOrganizationRepositoryError(
                    "adoption_identity_collision",
                    "portable resource identities collide within one domain",
                )
        return _ResourceScan(
            final_ids,
            merges,
            rows_by_domain,
            frozenset(deleted),
            sync_aliases,
        )

    def _resource_intents(
        self,
        resource_scan: _ResourceScan,
    ) -> tuple[_Intent, ...]:
        intents: list[_Intent] = []
        for row in resource_scan.rows["notes.keyword"]:
            local_id = str(row["id"])
            key = ("notes.keyword", local_id)
            if key not in resource_scan.final_ids:
                continue
            intents.append(
                _Intent(
                    "resources",
                    "notes.keyword",
                    resource_scan.final_ids[key],
                    "upsert",
                    {"keyword": str(row["name"])},
                    int(row["version"]),
                    self._merge_for(
                        "notes.keyword", local_id, resource_scan.merges
                    ),
                )
            )
        for domain in ("notes.keyword_collection", "notes.folder"):
            for row in resource_scan.rows[domain]:
                local_id = str(row["id"])
                key = (domain, local_id)
                if key not in resource_scan.final_ids:
                    continue
                parent_id = (
                    None
                    if row["parent_id"] is None
                    else resource_scan.final_ids[(domain, str(row["parent_id"]))]
                )
                intents.append(
                    _Intent(
                        "resources",
                        domain,
                        resource_scan.final_ids[key],
                        "upsert",
                        {"name": str(row["name"]), "parent_sync_id": parent_id},
                        int(row["version"]),
                        self._merge_for(domain, local_id, resource_scan.merges),
                    )
                )
        return tuple(intents)

    @staticmethod
    def _merge_for(
        domain: str,
        local_id: str,
        merges: Mapping[tuple[str, str], str],
    ) -> tuple[str, str] | None:
        remote = merges.get((domain, local_id))
        return None if remote is None else (local_id, remote)

    def _link_intents(
        self,
        cursor: sqlite3.Cursor,
        final_ids: Mapping[tuple[str, str], str],
        sync_aliases: Mapping[tuple[str, str], str],
    ) -> tuple[
        tuple[_Intent, ...],
        tuple[_Intent, ...],
        tuple[tuple[str, str, str], ...],
        list[dict[str, object]],
    ]:
        upserts: dict[tuple[str, str], _Intent] = {}
        tombstone_keys: set[tuple[str, str]] = set()
        forced_tombstones: set[tuple[str, str]] = set()
        active_keys: set[tuple[str, str]] = set()
        skipped: set[tuple[str, str, str]] = set()
        evidence: list[dict[str, object]] = []
        published_folder_ids = {
            sync_id
            for (domain, _local_id), sync_id in final_ids.items()
            if domain == "notes.folder"
        }

        def add(
            domain: str,
            payload: dict[str, object],
            source_version: int,
            *,
            deleted_evidence: bool = False,
            force_tombstone: bool = False,
        ) -> None:
            dependency = self._missing_dependency(domain, payload)
            if dependency is not None:
                skipped.add((domain, *dependency))
                evidence.append(
                    {"domain": domain, "payload": payload, "missing": dependency}
                )
                return
            members = _link_members(domain, payload)
            object_id = organization_link_id(domain, members)
            validate_organization_object_id(domain, object_id, payload)
            key = (domain, object_id)
            prior = upserts.get(key)
            if prior is None or source_version > prior.source_version:
                upserts[key] = _Intent(
                    "links", domain, object_id, "upsert", payload, source_version
                )
            if deleted_evidence:
                tombstone_keys.add(key)
                if force_tombstone:
                    forced_tombstones.add(key)
            else:
                active_keys.add(key)
            evidence.append(
                {
                    "domain": domain,
                    "object_id": object_id,
                    "deleted": deleted_evidence,
                    "source_version": source_version,
                }
            )

        for table, subject_type, subject_column in (
            ("note_keywords", "note", "note_id"),
            ("conversation_keywords", "conversation", "conversation_id"),
        ):
            rows = cursor.execute(
                f"SELECT link.{subject_column} AS subject_id, link.keyword_id, "
                f"keyword.version FROM {table} AS link JOIN keywords AS keyword "
                "ON keyword.id = link.keyword_id ORDER BY subject_id, link.keyword_id"
            ).fetchall()
            for row in rows:
                keyword_sync_id = final_ids.get(
                    ("notes.keyword", str(row["keyword_id"]))
                )
                if keyword_sync_id is None:
                    continue
                add(
                    "notes.keyword_link",
                    {
                        "subject_type": subject_type,
                        "subject_id": str(row["subject_id"]),
                        "keyword_sync_id": keyword_sync_id,
                    },
                    max(1, int(row["version"])),
                )
        rows = cursor.execute(
            "SELECT collection_id, keyword_id FROM collection_keywords "
            "ORDER BY collection_id, keyword_id"
        ).fetchall()
        for row in rows:
            collection_sync_id = final_ids.get(
                ("notes.keyword_collection", str(row["collection_id"]))
            )
            keyword_sync_id = final_ids.get(
                ("notes.keyword", str(row["keyword_id"]))
            )
            if collection_sync_id is None or keyword_sync_id is None:
                continue
            add(
                "notes.keyword_collection_link",
                {
                    "collection_sync_id": collection_sync_id,
                    "keyword_sync_id": keyword_sync_id,
                },
                1,
            )
        rows = cursor.execute(
            "SELECT folder_id, note_id, version, deleted FROM note_folder_memberships "
            "ORDER BY folder_id, note_id, id"
        ).fetchall()
        for row in rows:
            folder_sync_id = final_ids.get(
                ("notes.folder", str(row["folder_id"]))
            )
            if folder_sync_id is None:
                continue
            add(
                "notes.folder_link",
                {
                    "note_id": str(row["note_id"]),
                    "folder_sync_id": folder_sync_id,
                },
                int(row["version"]),
                deleted_evidence=bool(row["deleted"]),
            )
        suppressions = cursor.execute(
            "SELECT note_id, folder_sync_id FROM note_folder_sync_suppressions "
            "ORDER BY note_id, folder_sync_id"
        ).fetchall()
        for row in suppressions:
            folder_sync_id = sync_aliases.get(
                ("notes.folder", str(row["folder_sync_id"]))
            )
            if folder_sync_id is None or folder_sync_id not in published_folder_ids:
                continue
            add(
                "notes.folder_link",
                {
                    "note_id": str(row["note_id"]),
                    "folder_sync_id": folder_sync_id,
                },
                1,
                deleted_evidence=True,
                force_tombstone=True,
            )
        self._add_sync_log_deletions(cursor, final_ids, add)

        deleted_only = (tombstone_keys - active_keys) | forced_tombstones
        links = tuple(sorted(upserts.values(), key=lambda item: item.key))
        tombstones = tuple(
            _Intent(
                "tombstones",
                upserts[key].domain,
                upserts[key].object_id,
                "tombstone",
                upserts[key].payload,
                upserts[key].source_version,
            )
            for key in sorted(deleted_only)
        )
        evidence.sort(key=lambda item: _canonical_json(item))
        return links, tombstones, tuple(sorted(skipped)), evidence

    def _add_sync_log_deletions(
        self,
        cursor: sqlite3.Cursor,
        final_ids: Mapping[tuple[str, str], str],
        add: Callable[..., None],
    ) -> None:
        rows = cursor.execute(
            "SELECT change_id, entity, payload FROM sync_log WHERE operation = 'delete' "
            "AND entity IN ('note_keywords', 'conversation_keywords', "
            "'collection_keywords') ORDER BY change_id"
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(str(row["payload"]))
                entity = str(row["entity"])
                if entity == "collection_keywords":
                    wire = {
                        "collection_sync_id": final_ids[
                            (
                                "notes.keyword_collection",
                                str(payload["collection_id"]),
                            )
                        ],
                        "keyword_sync_id": final_ids[
                            ("notes.keyword", str(payload["keyword_id"]))
                        ],
                    }
                    domain = "notes.keyword_collection_link"
                else:
                    subject_type = "note" if entity == "note_keywords" else "conversation"
                    subject_key = "note_id" if subject_type == "note" else "conversation_id"
                    wire = {
                        "subject_type": subject_type,
                        "subject_id": str(payload[subject_key]),
                        "keyword_sync_id": final_ids[
                            ("notes.keyword", str(payload["keyword_id"]))
                        ],
                    }
                    domain = "notes.keyword_link"
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            add(
                domain,
                wire,
                max(1, int(row["change_id"])),
                deleted_evidence=True,
            )

    def _missing_dependency(
        self, domain: str, payload: Mapping[str, object]
    ) -> tuple[str, str] | None:
        if domain == "notes.folder_link":
            note_id = str(payload["note_id"])
            if note_id not in self.enrolled_note_ids:
                return ("notes.note", note_id)
        elif domain == "notes.keyword_link":
            subject_id = str(payload["subject_id"])
            if payload["subject_type"] == "note":
                if subject_id not in self.enrolled_note_ids:
                    return ("notes.note", subject_id)
            elif subject_id not in self.enrolled_conversation_ids:
                return ("chat.conversation", subject_id)
        return None

    @staticmethod
    def _token(baseline: str, key: str | None) -> str:
        return _canonical_json({"baseline": baseline, "key": key})

    @staticmethod
    def _parse_token(raw: object) -> tuple[str, str | None]:
        try:
            token = json.loads(str(raw))
            baseline, key = token["baseline"], token["key"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise NotesOrganizationRepositoryError(
                "invalid_inventory_checkpoint", "inventory checkpoint is malformed"
            ) from None
        if not isinstance(baseline, str) or len(baseline) != 64 or (
            key is not None and not isinstance(key, str)
        ):
            raise NotesOrganizationRepositoryError(
                "invalid_inventory_checkpoint", "inventory checkpoint is malformed"
            )
        return baseline, key

    @staticmethod
    def _next_entry(entries: tuple[_Intent, ...], last_key: str | None) -> _Intent | None:
        if last_key is None:
            return entries[0] if entries else None
        for index, entry in enumerate(entries):
            if entry.key == last_key:
                return entries[index + 1] if index + 1 < len(entries) else None
        raise NotesOrganizationRepositoryError(
            "invalid_inventory_checkpoint", "inventory key is not in its source baseline"
        )

    @staticmethod
    def _next_phase(phase: str) -> str:
        try:
            return (*_PHASES, "complete")[_PHASES.index(phase) + 1]
        except (ValueError, IndexError):
            raise NotesOrganizationRepositoryError(
                "invalid_inventory_checkpoint", "inventory phase is invalid"
            ) from None


def _parent_first(rows: Iterable[sqlite3.Row]) -> tuple[sqlite3.Row, ...]:
    by_id: dict[str, sqlite3.Row] = {}
    parent_ids: dict[str, str | None] = {}
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for row in rows:
        local_id = str(row["id"])
        if local_id in by_id:
            raise NotesOrganizationRepositoryError(
                "hierarchy_cycle", "legacy organization hierarchy contains a cycle"
            )
        parent_id = None if row["parent_id"] is None else str(row["parent_id"])
        by_id[local_id] = row
        parent_ids[local_id] = parent_id
        if parent_id is None:
            roots.append(local_id)
        else:
            children.setdefault(parent_id, []).append(local_id)

    roots.sort()
    for child_ids in children.values():
        child_ids.sort()
    ready = deque(roots)
    ordered: list[sqlite3.Row] = []
    while ready:
        local_id = ready.popleft()
        ordered.append(by_id[local_id])
        ready.extend(children.get(local_id, ()))
    if len(ordered) != len(by_id) or any(
        parent_id is not None and parent_id not in by_id
        for parent_id in parent_ids.values()
    ):
        raise NotesOrganizationRepositoryError(
            "hierarchy_cycle", "legacy organization hierarchy contains a cycle"
        )
    return tuple(ordered)


def _link_members(domain: str, payload: Mapping[str, object]) -> tuple[str, ...]:
    if domain == "notes.keyword_link":
        return (
            str(payload["subject_type"]),
            str(payload["subject_id"]),
            str(payload["keyword_sync_id"]),
        )
    if domain == "notes.keyword_collection_link":
        return (
            str(payload["collection_sync_id"]),
            str(payload["keyword_sync_id"]),
        )
    return (str(payload["note_id"]), str(payload["folder_sync_id"]))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


__all__ = ["InventoryRunResult", "LegacyNotesOrganizationInventory"]
