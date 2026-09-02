"""Crash-safe projection of Notes organization intents into SyncState."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.agent_lessons import (
    AgentLessonsSeedResult,
    initialize_agent_lessons_folder,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_organization_repository import (
    NotesOrganizationRepository,
)

from tldw_chatbook.Sync_Interop.notes_organization import (
    NOTES_ORGANIZATION_DOMAINS,
    new_organization_sync_id,
    organization_link_id,
)
from tldw_chatbook.Sync_Interop.notes_outbox_producer import NotesSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.sync_state import is_local_first_sync_profile_mode

_RESOURCE_SYNC_ID_TABLES = ("keywords", "keyword_collections")


class NotesOrganizationSyncService:
    """Copy immutable ChaChaNotes organization intents into the general outbox."""

    def __init__(
        self,
        *,
        notes_repository: Any,
        state_repository: Any,
        notes_producer: NotesSyncV2OutboxProducer | None = None,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.notes_repository = notes_repository
        self.state_repository = state_repository
        self.notes_producer = notes_producer
        self.failure_injector = failure_injector

    def for_notes_db(self, notes_db: Any) -> "NotesOrganizationSyncService":
        """Return an immutable view bound to the mutation's exact Notes owner."""

        if self.notes_repository.db is notes_db:
            return self
        producer = self.notes_producer
        if producer is not None:
            producer = NotesSyncV2OutboxProducer(
                state_repository=producer.state_repository,
                dataset_keys=producer.dataset_keys,
                notes_db=notes_db,
            )
        return type(self)(
            notes_repository=NotesOrganizationRepository(
                notes_db,
                server_profile_id=self.notes_repository.server_profile_id,
            ),
            state_repository=self.state_repository,
            notes_producer=producer,
            failure_injector=self.failure_injector,
        )

    def for_server_profile(
        self, server_profile_id: str
    ) -> "NotesOrganizationSyncService":
        """Return a view whose Notes repository is bound to the active profile."""

        profile_id = str(server_profile_id or "").strip()
        if not profile_id:
            raise ValueError("server_profile_id is required")
        if self.notes_repository.server_profile_id == profile_id:
            return self
        return type(self)(
            notes_repository=NotesOrganizationRepository(
                self.notes_repository.db,
                server_profile_id=profile_id,
            ),
            state_repository=self.state_repository,
            notes_producer=self.notes_producer,
            failure_injector=self.failure_injector,
        )

    def resolve_profile_scope(self, explicit: Any = None) -> dict[str, Any] | None:
        """Resolve the sole eligible implicit scope or validate an explicit one."""

        if explicit is None:
            profiles = [
                profile
                for profile in self.state_repository.list_sync_v2_profile_states()
                if is_local_first_sync_profile_mode(profile.get("profile_mode"))
            ]
            if not profiles:
                return None
            if len(profiles) > 1:
                raise ValueError(
                    "multiple eligible Notes profile scopes require explicit routing"
                )
            profile = profiles[0]
            return {
                "server_profile_id": profile["server_profile_id"],
                "authenticated_principal_id": profile["authenticated_principal_id"],
                "workspace_scope": profile["workspace_scope"],
            }
        required = {
            "server_profile_id",
            "authenticated_principal_id",
            "workspace_scope",
        }
        if not isinstance(explicit, dict) or set(explicit) != required:
            raise ValueError("invalid Notes profile scope mapping")
        scope = dict(explicit)
        self._profile(**scope)
        return scope

    def create_folder(
        self,
        *,
        folder_repository: Any,
        name: str,
        parent_id: str | None,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> Any:
        """Atomically create one ready-state folder and its publication intent."""

        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None:
            raise ValueError("persisted Notes profile scope is required")
        if profile.get("profile_mode") == "local_only":
            return folder_repository.create_folder(name=name, parent_id=parent_id)
        dataset_id = str(profile.get("dataset_id") or "")
        if not dataset_id:
            raise ValueError("synchronized profile is missing dataset identity")
        with self.notes_repository.db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
            )
            parent_sync_id = None
            if parent_id is not None:
                parent = cursor.execute(
                    "SELECT sync_id FROM note_folders WHERE id = ? AND deleted = 0",
                    (parent_id,),
                ).fetchone()
                if parent is None or not parent["sync_id"]:
                    raise ValueError("parent folder is not synchronized")
                parent_sync_id = str(parent["sync_id"])
            folder = folder_repository.create_folder(
                name=name,
                parent_id=parent_id,
                cursor=cursor,
            )
            persisted = cursor.execute(
                "SELECT sync_id FROM note_folders WHERE id = ?",
                (folder.folder_id,),
            ).fetchone()
            if persisted is None or not persisted["sync_id"]:
                raise ValueError("created folder is not synchronized")
            sync_id = str(persisted["sync_id"])
            self._record_resource(
                cursor,
                profile=server_profile_id,
                dataset=dataset_id,
                domain="notes.folder",
                object_id=sync_id,
                operation="upsert",
                payload={"name": folder.name, "parent_sync_id": parent_sync_id},
                source_version=folder.version,
            )
        return folder

    def attach_folder_link(
        self,
        *,
        folder_repository: Any,
        folder_id: str,
        note_id: str,
        **scope: Any,
    ) -> Any:
        """Attach one manual placement and publish only a newly effective link."""

        profile = self._profile(**scope)
        if profile is None:
            return folder_repository.attach_manual(folder_id=folder_id, note_id=note_id)
        dataset_id = self._dataset_id(profile)
        with self.notes_repository.db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            was_effective = self._folder_link_effective(cursor, folder_id, note_id)
            folder = cursor.execute(
                "SELECT sync_id FROM note_folders WHERE id = ?", (folder_id,)
            ).fetchone()
            if folder is None or not folder["sync_id"]:
                raise ValueError("folder is not synchronized")
            cursor.execute(
                "DELETE FROM note_folder_sync_suppressions "
                "WHERE note_id = ? AND folder_sync_id = ?",
                (note_id, str(folder["sync_id"])),
            )
            result = folder_repository.attach_manual(
                folder_id=folder_id, note_id=note_id, cursor=cursor
            )
            if not was_effective:
                self._record_folder_link(
                    cursor,
                    folder_id=folder_id,
                    note_id=note_id,
                    operation="upsert",
                    profile=scope["server_profile_id"],
                    dataset=dataset_id,
                )
        return result

    def detach_folder_link(
        self,
        *,
        folder_repository: Any,
        folder_id: str,
        note_id: str,
        expected_version: int,
        **scope: Any,
    ) -> bool:
        """Detach manual provenance and tombstone only an absent effective union."""

        profile = self._profile(**scope)
        if profile is None:
            return folder_repository.detach_manual(
                folder_id=folder_id,
                note_id=note_id,
                expected_version=expected_version,
            )
        dataset_id = self._dataset_id(profile)
        with self.notes_repository.db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            was_effective = self._folder_link_effective(cursor, folder_id, note_id)
            result = folder_repository.detach_manual(
                folder_id=folder_id,
                note_id=note_id,
                expected_version=expected_version,
                cursor=cursor,
            )
            if (
                result
                and was_effective
                and not self._folder_link_effective(cursor, folder_id, note_id)
            ):
                self._record_folder_link(
                    cursor,
                    folder_id=folder_id,
                    note_id=note_id,
                    operation="tombstone",
                    profile=scope["server_profile_id"],
                    dataset=dataset_id,
                )
        return result

    def mutate_managed_folder_links(
        self,
        *,
        folder_repository: Any,
        mutation_method: str,
        owner_id: str,
        desired: Sequence[tuple[str, str]] = (),
        **scope: Any,
    ) -> Any:
        """Run a managed-owner mutation and journal only effective-union changes."""

        mutate = getattr(folder_repository, mutation_method)
        arguments: dict[str, Any] = {"owner_id": owner_id}
        if mutation_method == "reconcile_managed":
            arguments["desired"] = tuple(desired)
        profile = self._profile(**scope)
        if profile is None:
            return mutate(**arguments)
        dataset_id = self._dataset_id(profile)
        with self.notes_repository.db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            before = self._effective_folder_links(cursor)
            result = mutate(cursor=cursor, **arguments)
            after = self._effective_folder_links(cursor)
            for folder_id, note_id in sorted(before ^ after):
                self._record_folder_link(
                    cursor,
                    folder_id=folder_id,
                    note_id=note_id,
                    operation="upsert"
                    if (folder_id, note_id) in after
                    else "tombstone",
                    profile=scope["server_profile_id"],
                    dataset=dataset_id,
                )
        return result

    def sync_subject_keywords(
        self,
        *,
        subject_type: str,
        subject_id: str,
        keywords: Sequence[str],
        notes_db: Any = None,
        cursor: Any = None,
        **scope: Any,
    ) -> list[str] | None:
        """Replace a note/conversation keyword set in one ready Notes transaction."""

        profile = self._profile(**scope)
        if profile is None:
            return None
        if subject_type not in {"note", "conversation"}:
            raise ValueError("subject_type must be note or conversation")
        dataset_id = self._dataset_id(profile)
        service = self.for_notes_db(notes_db) if notes_db is not None else self
        if service is not self:
            return service.sync_subject_keywords(
                subject_type=subject_type,
                subject_id=subject_id,
                keywords=keywords,
                cursor=cursor,
                **scope,
            )
        db = self.notes_repository.db
        requested: dict[str, str] = {}
        for value in keywords:
            text = value.strip()
            if text:
                requested.setdefault(text.casefold(), text)
        normalized = list(requested.values())
        link_table = (
            "note_keywords" if subject_type == "note" else "conversation_keywords"
        )
        subject_column = "note_id" if subject_type == "note" else "conversation_id"
        link = (
            db.link_note_to_keyword
            if subject_type == "note"
            else db.link_conversation_to_keyword
        )
        unlink = (
            db.unlink_note_from_keyword
            if subject_type == "note"
            else db.unlink_conversation_from_keyword
        )

        def mutate(owner_cursor: Any) -> None:
            self._require_ready(
                owner_cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            existing = {
                str(row["keyword"]).casefold(): row
                for row in owner_cursor.execute(
                    f"SELECT k.* FROM keywords k JOIN {link_table} l ON l.keyword_id = k.id WHERE l.{subject_column} = ? AND k.deleted = 0",
                    (subject_id,),
                ).fetchall()
            }
            for key, text in requested.items():
                if key in existing:
                    continue
                keyword = owner_cursor.execute(
                    "SELECT * FROM keywords WHERE keyword = ? COLLATE NOCASE", (text,)
                ).fetchone()
                created = keyword is None or bool(keyword["deleted"])
                if created:
                    keyword_id = db.add_keyword(text, cursor=owner_cursor)
                    keyword = owner_cursor.execute(
                        "SELECT * FROM keywords WHERE id = ?", (keyword_id,)
                    ).fetchone()
                keyword_sync_id = self._resource_sync_id(
                    owner_cursor, "keywords", int(keyword["id"])
                )
                if created:
                    self._record_resource(
                        owner_cursor,
                        domain="notes.keyword",
                        object_id=keyword_sync_id,
                        operation="upsert",
                        payload={"keyword": str(keyword["keyword"])},
                        source_version=int(keyword["version"]),
                        profile=scope["server_profile_id"],
                        dataset=dataset_id,
                    )
                if link(subject_id, int(keyword["id"]), cursor=owner_cursor):
                    self._record_keyword_link(
                        owner_cursor,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        keyword_sync_id=keyword_sync_id,
                        operation="upsert",
                        profile=scope["server_profile_id"],
                        dataset=dataset_id,
                    )
            for key, keyword in existing.items():
                if key in requested:
                    continue
                if unlink(subject_id, int(keyword["id"]), cursor=owner_cursor):
                    self._record_keyword_link(
                        owner_cursor,
                        subject_type=subject_type,
                        subject_id=subject_id,
                        keyword_sync_id=self._required_sync_id(keyword),
                        operation="tombstone",
                        profile=scope["server_profile_id"],
                        dataset=dataset_id,
                    )

        if cursor is None:
            with db.transaction() as owner_cursor:
                mutate(owner_cursor)
        else:
            mutate(cursor)
        if self.failure_injector is not None:
            self.failure_injector("after_notes_mutation_and_intent")
        return normalized

    def create_keyword(self, *, keyword: str, **scope: Any) -> int | None:
        """Create one portable keyword after complete group readiness."""

        profile = self._profile(**scope)
        db = self.notes_repository.db
        if profile is None:
            return db.add_keyword(keyword)
        dataset_id = self._dataset_id(profile)
        with db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            keyword_id = db.add_keyword(keyword, cursor=cursor)
            if keyword_id is None:
                return None
            row = cursor.execute(
                "SELECT * FROM keywords WHERE id = ?", (keyword_id,)
            ).fetchone()
            sync_id = self._resource_sync_id(cursor, "keywords", keyword_id)
            self._record_resource(
                cursor,
                domain="notes.keyword",
                object_id=sync_id,
                operation="upsert",
                payload={"keyword": str(row["keyword"])},
                source_version=int(row["version"]),
                profile=scope["server_profile_id"],
                dataset=dataset_id,
            )
        return keyword_id

    def mutate_keyword(
        self,
        *,
        keyword_id: int,
        expected_version: int,
        keyword: str | None = None,
        delete: bool = False,
        **scope: Any,
    ) -> bool:
        """Rename or tombstone a keyword in the mutation+intent transaction."""

        profile = self._profile(**scope)
        db = self.notes_repository.db
        if delete:
            mutate = db.soft_delete_keyword
            arguments: dict[str, Any] = {"expected_version": expected_version}
        else:
            if keyword is None:
                raise ValueError("keyword is required for an update")
            mutate = db.update_keyword
            arguments = {"keyword_text": keyword, "expected_version": expected_version}
        if profile is None:
            return bool(mutate(keyword_id, **arguments))
        dataset_id = self._dataset_id(profile)
        with db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            before = cursor.execute(
                "SELECT sync_id FROM keywords WHERE id = ?", (keyword_id,)
            ).fetchone()
            if before is None or not before["sync_id"]:
                raise ValueError("keyword is not synchronized")
            result = bool(mutate(keyword_id, cursor=cursor, **arguments))
            if not result:
                return False
            row = cursor.execute(
                "SELECT * FROM keywords WHERE id = ?", (keyword_id,)
            ).fetchone()
            self._record_resource(
                cursor,
                domain="notes.keyword",
                object_id=str(before["sync_id"]),
                operation="tombstone" if delete else "upsert",
                payload={} if delete else {"keyword": str(row["keyword"])},
                source_version=int(row["version"]),
                profile=scope["server_profile_id"],
                dataset=dataset_id,
            )
        return result

    def create_keyword_collection(
        self, *, name: str, parent_id: int | None = None, **scope: Any
    ) -> int | None:
        """Create a portable keyword collection after group readiness."""

        profile = self._profile(**scope)
        db = self.notes_repository.db
        if profile is None:
            return db.add_keyword_collection(name, parent_id)
        dataset_id = self._dataset_id(profile)
        with db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            collection_id = db.add_keyword_collection(name, parent_id, cursor=cursor)
            if collection_id is None:
                return None
            row = cursor.execute(
                "SELECT * FROM keyword_collections WHERE id = ?", (collection_id,)
            ).fetchone()
            sync_id = self._resource_sync_id(
                cursor, "keyword_collections", collection_id
            )
            parent_sync_id = None
            if row["parent_id"] is not None:
                parent = cursor.execute(
                    "SELECT sync_id FROM keyword_collections WHERE id = ? AND deleted = 0",
                    (row["parent_id"],),
                ).fetchone()
                if parent is None or not parent["sync_id"]:
                    raise ValueError("parent collection is not synchronized")
                parent_sync_id = str(parent["sync_id"])
            self._record_resource(
                cursor,
                domain="notes.keyword_collection",
                object_id=sync_id,
                operation="upsert",
                payload={"name": str(row["name"]), "parent_sync_id": parent_sync_id},
                source_version=int(row["version"]),
                profile=scope["server_profile_id"],
                dataset=dataset_id,
            )
        return collection_id

    def mutate_keyword_collection(
        self,
        *,
        collection_id: int,
        expected_version: int,
        update_data: dict[str, Any] | None = None,
        delete: bool = False,
        **scope: Any,
    ) -> bool:
        """Update or tombstone one collection with its immutable intent."""

        profile = self._profile(**scope)
        db = self.notes_repository.db
        mutate = (
            db.soft_delete_keyword_collection
            if delete
            else db.update_keyword_collection
        )
        arguments: dict[str, Any] = {"expected_version": expected_version}
        if not delete:
            arguments["update_data"] = update_data or {}
        if profile is None:
            return bool(mutate(collection_id, **arguments))
        dataset_id = self._dataset_id(profile)
        with db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            before = cursor.execute(
                "SELECT sync_id FROM keyword_collections WHERE id = ?", (collection_id,)
            ).fetchone()
            if before is None or not before["sync_id"]:
                raise ValueError("collection is not synchronized")
            result = bool(mutate(collection_id, cursor=cursor, **arguments))
            if not result:
                return False
            row = cursor.execute(
                "SELECT * FROM keyword_collections WHERE id = ?", (collection_id,)
            ).fetchone()
            payload: dict[str, object] = {}
            if not delete:
                parent_sync_id = None
                if row["parent_id"] is not None:
                    parent = cursor.execute(
                        "SELECT sync_id FROM keyword_collections WHERE id = ?",
                        (row["parent_id"],),
                    ).fetchone()
                    if parent is None or not parent["sync_id"]:
                        raise ValueError("parent collection is not synchronized")
                    parent_sync_id = str(parent["sync_id"])
                payload = {"name": str(row["name"]), "parent_sync_id": parent_sync_id}
            self._record_resource(
                cursor,
                domain="notes.keyword_collection",
                object_id=str(before["sync_id"]),
                operation="tombstone" if delete else "upsert",
                payload=payload,
                source_version=int(row["version"]),
                profile=scope["server_profile_id"],
                dataset=dataset_id,
            )
        return result

    def set_collection_keyword_link(
        self,
        *,
        collection_id: int,
        keyword_id: int,
        linked: bool,
        **scope: Any,
    ) -> bool:
        """Mutate one collection-keyword link and journal its relationship."""

        profile = self._profile(**scope)
        db = self.notes_repository.db
        mutate = (
            db.link_collection_to_keyword
            if linked
            else db.unlink_collection_from_keyword
        )
        if profile is None:
            return bool(mutate(collection_id, keyword_id))
        dataset_id = self._dataset_id(profile)
        with db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=scope["server_profile_id"],
                dataset_id=dataset_id,
            )
            collection = cursor.execute(
                "SELECT sync_id FROM keyword_collections WHERE id = ? AND deleted = 0",
                (collection_id,),
            ).fetchone()
            keyword = cursor.execute(
                "SELECT sync_id FROM keywords WHERE id = ? AND deleted = 0",
                (keyword_id,),
            ).fetchone()
            if (
                collection is None
                or keyword is None
                or not collection["sync_id"]
                or not keyword["sync_id"]
            ):
                raise ValueError("collection link resources are not synchronized")
            result = bool(mutate(collection_id, keyword_id, cursor=cursor))
            if result:
                payload: dict[str, object] = {
                    "collection_sync_id": str(collection["sync_id"]),
                    "keyword_sync_id": str(keyword["sync_id"]),
                }
                object_id = organization_link_id(
                    "notes.keyword_collection_link",
                    (
                        str(collection["sync_id"]),
                        str(keyword["sync_id"]),
                    ),
                )
                self._record_link(
                    cursor,
                    domain="notes.keyword_collection_link",
                    object_id=object_id,
                    operation="upsert" if linked else "tombstone",
                    payload=payload,
                    profile=scope["server_profile_id"],
                    dataset=dataset_id,
                )
        return result

    def _profile(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> dict[str, Any] | None:
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None:
            raise ValueError("persisted Notes profile scope is required")
        if profile.get("profile_mode") == "local_only":
            return None
        return profile

    def pending_agent_lesson_content_scope(
        self,
        *,
        content_only: bool = False,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> dict[str, str]:
        """Validate the later pending-content path without permitting organization.

        TASK-24308 may use the returned identities for its separate note-plus-
        receipt transaction. This hook performs no write and stays closed unless
        the caller explicitly proves it is requesting content-only pending work.
        """

        if content_only is not True:
            raise ValueError("pending Agent Lessons hook is content-only")
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        if profile is None or profile.get("profile_mode") == "local_only":
            raise ValueError("pending Agent Lessons requires a synchronized profile")
        return {
            "server_profile_id": server_profile_id,
            "dataset_id": self._dataset_id(profile),
        }

    def notes_organization_ready(
        self, *, server_profile_id: str, dataset_id: str
    ) -> bool:
        """Return whether every durable local and server readiness gate is closed."""

        connection = self.notes_repository.db.get_connection()
        checkpoint = connection.execute(
            "SELECT local_state, server_state, inventory_phase, error_code "
            "FROM notes_organization_sync_checkpoints WHERE server_profile_id = ? "
            "AND dataset_id = ?",
            (server_profile_id, dataset_id),
        ).fetchone()
        if checkpoint is None:
            return False
        open_review = connection.execute(
            "SELECT 1 FROM notes_organization_adoption_reviews AS review "
            "WHERE review.server_profile_id = ? AND review.dataset_id = ? "
            "AND review.state = 'open' AND NOT EXISTS (SELECT 1 FROM "
            "note_organization_receipts AS receipt WHERE receipt.review_id = "
            "review.review_id AND receipt.state = 'placement_review') LIMIT 1",
            (server_profile_id, dataset_id),
        ).fetchone()
        return (
            checkpoint["local_state"] == "ready"
            and checkpoint["server_state"] == "ready"
            and checkpoint["inventory_phase"] == "complete"
            and checkpoint["error_code"] is None
            and open_review is None
        )

    def initialize_agent_lessons_seed(
        self, *, server_profile_id: str, dataset_id: str
    ) -> AgentLessonsSeedResult:
        """Seed the conventional root only after complete group readiness."""

        if not self.notes_organization_ready(
            server_profile_id=server_profile_id, dataset_id=dataset_id
        ):
            return AgentLessonsSeedResult("not_ready")
        repository = NotesOrganizationRepository(
            self.notes_repository.db, server_profile_id=server_profile_id
        )
        return initialize_agent_lessons_folder(
            repository.db,
            scope_mode="synchronized",
            profile_id=server_profile_id,
            dataset_id=dataset_id,
            organization_repository=repository,
        )

    def _agent_lessons_seed_is_unknown(
        self, *, server_profile_id: str, dataset_id: str
    ) -> bool:
        row = self.notes_repository.db.get_connection().execute(
            "SELECT state FROM agent_lessons_seed_state WHERE profile_id = ? "
            "AND dataset_id = ?",
            (server_profile_id, dataset_id),
        ).fetchone()
        return row is None or row["state"] == "unknown"

    async def advance_enrollment(
        self,
        *,
        server_service: Any,
        local_first_service: Any,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        display_name: str,
        enrolled_note_ids: set[str],
        enrolled_conversation_ids: set[str],
        after_checkpoint: Callable[[str, str | None], None] | None = None,
    ) -> dict[str, Any]:
        """Resume server bootstrap, history pull, adoption, and legacy inventory."""

        from tldw_chatbook.Sync_Interop.notes_organization_inventory import (
            LegacyNotesOrganizationInventory,
        )

        def noop_checkpoint(_stage: str, _cursor: str | None = None) -> None:
            return None

        callback = after_checkpoint or noop_checkpoint
        profile = await server_service.bootstrap_notes_organization_profile(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            display_name=display_name,
        )
        dataset = profile.get("dataset") or {}
        dataset_id = str(dataset.get("dataset_id") or "")
        status = dataset.get("notes_organization") or {}
        if not dataset_id or not isinstance(status, dict):
            raise ValueError("Notes organization server status is required")
        server_state = str(status.get("state") or "failed")
        persisted_profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        metadata = (
            persisted_profile.get("dry_run_metadata")
            if persisted_profile is not None
            else {}
        )
        bootstrap_id = (
            metadata.get("notes_organization_bootstrap_identity")
            if isinstance(metadata, dict)
            else None
        )
        if not bootstrap_id:
            import uuid

            scope = ":".join(
                (
                    server_profile_id,
                    authenticated_principal_id or "",
                    workspace_scope or "",
                )
            )
            bootstrap_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"tldw:notes-organization:{scope}")
            )
        checkpoint = self._record_server_enrollment_status(
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
            bootstrap_id=str(bootstrap_id),
            state=server_state,
            captured_count=int(status.get("captured_count") or 0),
            expected_count=int(status.get("expected_count") or 0),
            error_code=status.get("error_code"),
        )
        callback("server_status", checkpoint["pull_cursor"])
        if server_state == "initializing":
            return {"status": "initializing", "dataset_id": dataset_id}
        if server_state != "ready":
            return {
                "status": "failed",
                "dataset_id": dataset_id,
                "error_code": status.get("error_code") or "bootstrap_failed",
            }
        if self.notes_organization_ready(
            server_profile_id=server_profile_id, dataset_id=dataset_id
        ) and not self._agent_lessons_seed_is_unknown(
            server_profile_id=server_profile_id, dataset_id=dataset_id
        ):
            seeded = self.initialize_agent_lessons_seed(
                server_profile_id=server_profile_id, dataset_id=dataset_id
            )
            if seeded.status == "adoption_review":
                return {"status": "adoption_review", "dataset_id": dataset_id}
            self.finalize_pending_note_organization_receipts(
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
            )
            return {"status": "ready", "dataset_id": dataset_id}

        pull = await local_first_service.pull_notes_organization_history(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        self._set_enrollment_local_state(
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
            local_state="adoption_review",
            pull_cursor=pull.get("next_cursor"),
            error_code=None,
        )
        callback("pull_complete", pull.get("next_cursor"))
        callback("adoption_review", pull.get("next_cursor"))
        open_review = (
            self.notes_repository.db.get_connection()
            .execute(
                "SELECT 1 FROM notes_organization_adoption_reviews WHERE "
                "server_profile_id = ? AND dataset_id = ? AND state = 'open' LIMIT 1",
                (server_profile_id, dataset_id),
            )
            .fetchone()
        )
        if open_review is not None or pull.get("conflicts"):
            return {"status": "adoption_review", "dataset_id": dataset_id}

        inventory = LegacyNotesOrganizationInventory(
            NotesOrganizationRepository(
                self.notes_repository.db, server_profile_id=server_profile_id
            ),
            dataset_id=dataset_id,
            enrolled_note_ids=enrolled_note_ids,
            enrolled_conversation_ids=enrolled_conversation_ids,
        ).run(after_commit=lambda phase, key: callback(f"inventory:{phase}", key))
        if inventory.status != "complete":
            return {
                "status": "adoption_review",
                "dataset_id": dataset_id,
                "error_code": inventory.reason_code,
            }
        if inventory.skipped_dependencies:
            self._set_enrollment_local_state(
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
                local_state="adoption_review",
                pull_cursor=pull.get("next_cursor"),
                error_code="notes_organization_dependency_missing",
            )
            return {
                "status": "adoption_review",
                "dataset_id": dataset_id,
                "error_code": "notes_organization_dependency_missing",
            }
        self._set_enrollment_local_state(
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
            local_state="ready",
            pull_cursor=pull.get("next_cursor"),
            error_code=None,
        )
        seeded = self.initialize_agent_lessons_seed(
            server_profile_id=server_profile_id, dataset_id=dataset_id
        )
        if seeded.status == "adoption_review":
            return {"status": "adoption_review", "dataset_id": dataset_id}
        self.finalize_pending_note_organization_receipts(
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
        )
        callback("ready", pull.get("next_cursor"))
        return {"status": "ready", "dataset_id": dataset_id}

    def resolve_placement_review(
        self, *, review_id: str, action: str, new_name: str | None = None
    ) -> bool:
        """Resolve a placement review with its mutation and intents atomically."""

        if action not in {"merge", "rename_local", "keep_local"}:
            raise ValueError("Unsupported Notes organization adoption action")
        db = self.notes_repository.db
        with db.transaction() as cursor:
            row = cursor.execute(
                "SELECT review.*, receipt.receipt_id FROM "
                "notes_organization_adoption_reviews AS review JOIN "
                "note_organization_receipts AS receipt ON receipt.review_id = "
                "review.review_id AND receipt.state = 'placement_review' "
                "WHERE review.review_id = ? AND review.state = 'open'",
                (review_id,),
            ).fetchone()
            if row is None:
                return False
            receipt = cursor.execute(
                "SELECT * FROM note_organization_receipts WHERE receipt_id = ?",
                (str(row["receipt_id"]),),
            ).fetchone()
            binding = self._receipt_request_binding(receipt)
            if binding is None:
                raise ValueError("Placement review has no valid publication scope")
            server_profile_id = str(binding.get("server_profile_id") or "")
            dataset_id = str(binding.get("dataset_id") or "")
            if (
                server_profile_id != self.notes_repository.server_profile_id
                or not dataset_id
                or str(row["server_profile_id"]) != server_profile_id
                or str(row["dataset_id"]) != dataset_id
            ):
                raise ValueError("Placement review scope does not match finalizer")
            if not self._ready_for_receipt_finalization(
                cursor,
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
            ):
                raise ValueError("Notes organization group is not ready")
            note = cursor.execute(
                "SELECT * FROM notes WHERE id = ? AND deleted = 0",
                (str(receipt["note_id"]),),
            ).fetchone()
            if note is None or int(note["version"]) != int(receipt["note_version"]):
                raise ValueError("Placement review note state changed")
            organization = db._library_organization_for_notes(
                cursor, [str(note["id"])]
            )[str(note["id"])]
            if str(organization["organization_version"]) != str(
                receipt["organization_version"]
            ):
                raise ValueError("Placement review organization state changed")
            if str(row["domain"]) != "notes.folder":
                raise ValueError("Placement review is not a folder review")
            folder_id = str(row["local_object_id"])
            if action == "merge":
                remote_object_id = str(row["remote_object_id"] or "").strip()
                folder = cursor.execute(
                    "SELECT sync_id FROM note_folders WHERE id = ? AND deleted = 0",
                    (folder_id,),
                ).fetchone()
                if folder is None or not folder["sync_id"]:
                    raise ValueError("Notes organization local resource is missing")
                if remote_object_id:
                    cursor.execute(
                        "UPDATE note_folders SET sync_id = ? WHERE id = ?",
                        (remote_object_id, folder_id),
                    )
            elif action == "rename_local":
                folder = cursor.execute(
                    "SELECT version FROM note_folders WHERE id = ? AND deleted = 0",
                    (folder_id,),
                ).fetchone()
                if folder is None:
                    raise ValueError("Notes organization local resource is missing")
                LocalNoteFolderRepository(db).rename_folder(
                    folder_id,
                    name=new_name,
                    expected_version=int(folder["version"]),
                    cursor=cursor,
                )
            if self.failure_injector is not None:
                self.failure_injector("after_placement_resolution_mutation")
            now = _utc_now()
            cursor.execute(
                "UPDATE notes_organization_adoption_reviews SET state = 'resolved', "
                "resolution = ?, resolved_at = ?, updated_at = ? WHERE review_id = ? "
                "AND state = 'open'",
                (action, now, now, review_id),
            )
            if not self._finalize_resolved_placement_receipt(
                cursor,
                receipt=receipt,
                note=note,
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
            ):
                raise ValueError("Placement review could not be finalized")
        return True

    def finalize_pending_note_organization_receipts(
        self, *, server_profile_id: str, dataset_id: str
    ) -> dict[str, int]:
        """Finalize every eligible receipt in its owning Notes transaction."""

        db = self.notes_repository.db
        counts = {"finalized": 0, "placement_review": 0, "cancelled": 0}
        receipts = db.get_connection().execute(
            "SELECT receipt_id FROM note_organization_receipts ORDER BY created_at, receipt_id"
        ).fetchall()
        for candidate in receipts:
            with db.transaction() as cursor:
                receipt = cursor.execute(
                    "SELECT * FROM note_organization_receipts WHERE receipt_id = ?",
                    (str(candidate["receipt_id"]),),
                ).fetchone()
                if receipt is None:
                    continue
                binding = self._receipt_request_binding(receipt)
                if (
                    binding is None
                    or binding.get("server_profile_id") != server_profile_id
                    or binding.get("dataset_id") != dataset_id
                ):
                    continue
                request_fingerprint = binding.get("fingerprint")
                if (
                    not isinstance(request_fingerprint, str)
                    or len(request_fingerprint) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in request_fingerprint
                    )
                ):
                    continue
                note = cursor.execute(
                    "SELECT * FROM notes WHERE id = ?", (str(receipt["note_id"]),)
                ).fetchone()
                if note is None or bool(note["deleted"]):
                    self._cancel_receipt_with_cursor(cursor, receipt)
                    counts["cancelled"] += 1
                    continue
                if int(note["version"]) != int(receipt["note_version"]):
                    continue
                organization = db._library_organization_for_notes(
                    cursor, [str(note["id"])]
                )[str(note["id"])]
                if (
                    str(organization["organization_version"])
                    != str(receipt["organization_version"])
                ):
                    continue
                if not self._ready_for_receipt_finalization(
                    cursor,
                    server_profile_id=server_profile_id,
                    dataset_id=dataset_id,
                ):
                    continue
                if str(receipt["state"]) == "placement_review":
                    if self._finalize_resolved_placement_receipt(
                        cursor,
                        receipt=receipt,
                        note=note,
                        server_profile_id=server_profile_id,
                        dataset_id=dataset_id,
                    ):
                        counts["finalized"] += 1
                    continue
                base_version = binding.get("expected_version")
                if int(note["version"]) == 1:
                    if base_version is not None:
                        continue
                elif (
                    not isinstance(base_version, int)
                    or isinstance(base_version, bool)
                    or base_version != int(note["version"]) - 1
                ):
                    continue
                keywords = NotesInteropService._receipt_requested_keywords(receipt)
                if NotesInteropService._keyword_identity_conflict(
                    cursor,
                    keywords,
                    profile=server_profile_id,
                    dataset=dataset_id,
                ):
                    continue
                self._finalize_pending_receipt_with_cursor(
                    cursor,
                    receipt=receipt,
                    note=note,
                    keywords=keywords,
                    base_version=base_version,
                    server_profile_id=server_profile_id,
                    dataset_id=dataset_id,
                    request_fingerprint=request_fingerprint,
                    counts=counts,
                )
        return counts

    @staticmethod
    def _receipt_request_binding(receipt: Any) -> dict[str, Any] | None:
        """Return the durable request binding, failing closed on malformed state."""

        try:
            stored = json.loads(str(receipt["requested_keywords_json"]))
            binding = stored[-1]["_request"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            return None
        return binding if isinstance(binding, dict) else None

    @staticmethod
    def _ready_for_receipt_finalization(
        cursor: Any, *, server_profile_id: str, dataset_id: str
    ) -> bool:
        checkpoint = cursor.execute(
            "SELECT local_state, server_state, inventory_phase, error_code FROM "
            "notes_organization_sync_checkpoints WHERE server_profile_id = ? AND dataset_id = ?",
            (server_profile_id, dataset_id),
        ).fetchone()
        if (
            checkpoint is None
            or checkpoint["local_state"] != "ready"
            or checkpoint["server_state"] != "ready"
            or checkpoint["inventory_phase"] != "complete"
            or checkpoint["error_code"] is not None
        ):
            return False
        return cursor.execute(
            "SELECT 1 FROM notes_organization_adoption_reviews AS review "
            "WHERE review.server_profile_id = ? AND review.dataset_id = ? "
            "AND review.state = 'open' AND NOT EXISTS (SELECT 1 FROM "
            "note_organization_receipts AS receipt WHERE receipt.review_id = "
            "review.review_id AND receipt.state = 'placement_review') LIMIT 1",
            (server_profile_id, dataset_id),
        ).fetchone() is None

    def _finalize_pending_receipt_with_cursor(
        self,
        cursor: Any,
        *,
        receipt: Any,
        note: Any,
        keywords: Sequence[str],
        base_version: int | None,
        server_profile_id: str,
        dataset_id: str,
        request_fingerprint: str,
        counts: dict[str, int],
    ) -> None:
        db = self.notes_repository.db
        repository = NotesOrganizationRepository(
            db, server_profile_id=server_profile_id
        )
        folders = LocalNoteFolderRepository(db)
        folder, folder_created, collisions = NotesInteropService._ensure_save_folder(
            cursor,
            folder=(
                str(receipt["requested_folder_name"])
                if receipt["requested_folder_name"] is not None
                else None
            ),
            folder_sync_id=(
                str(receipt["requested_folder_sync_id"])
                if receipt["requested_folder_sync_id"] is not None
                else None
            ),
        )
        if folder_created and folder is not None:
            NotesInteropService._record_organization_intent(
                repository,
                cursor,
                profile=server_profile_id,
                dataset=dataset_id,
                domain="notes.folder",
                object_id=str(folder["sync_id"]),
                payload={"name": str(folder["name"]), "parent_sync_id": None},
                source_version=int(folder["version"]),
            )

        for keyword in keywords:
            row = cursor.execute(
                "SELECT * FROM keywords WHERE keyword = ? COLLATE BINARY AND deleted = 0",
                (keyword,),
            ).fetchone()
            created = row is None
            if created:
                keyword_id = db.add_keyword(keyword, cursor=cursor)
                row = cursor.execute(
                    "SELECT * FROM keywords WHERE id = ?", (keyword_id,)
                ).fetchone()
                NotesInteropService._record_organization_intent(
                    repository,
                    cursor,
                    profile=server_profile_id,
                    dataset=dataset_id,
                    domain="notes.keyword",
                    object_id=str(row["sync_id"]),
                    payload={"keyword": str(row["keyword"])},
                    source_version=int(row["version"]),
                )
            if db.link_note_to_keyword(
                str(note["id"]), int(row["id"]), cursor=cursor
            ):
                NotesInteropService._record_organization_link_intent(
                    repository,
                    cursor,
                    profile=server_profile_id,
                    dataset=dataset_id,
                    domain="notes.keyword_link",
                    members=("note", str(note["id"]), str(row["sync_id"])),
                    payload={
                        "subject_type": "note",
                        "subject_id": str(note["id"]),
                        "keyword_sync_id": str(row["sync_id"]),
                    },
                )

        if folder is not None and not collisions:
            existed = cursor.execute(
                "SELECT 1 FROM note_folder_memberships WHERE folder_id = ? "
                "AND note_id = ? AND deleted = 0 AND ownership = 'manual'",
                (str(folder["id"]), str(note["id"])),
            ).fetchone()
            folders.attach_manual(
                folder_id=str(folder["id"]), note_id=str(note["id"]), cursor=cursor
            )
            if existed is None:
                NotesInteropService._record_organization_link_intent(
                    repository,
                    cursor,
                    profile=server_profile_id,
                    dataset=dataset_id,
                    domain="notes.folder_link",
                    members=(str(note["id"]), str(folder["sync_id"])),
                    payload={
                        "note_id": str(note["id"]),
                        "folder_sync_id": str(folder["sync_id"]),
                    },
                )

        self._insert_note_sync_intent(
            cursor,
            note,
            intent_id=str(receipt["receipt_id"]),
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
            base_version=base_version,
            request_fingerprint=request_fingerprint,
        )
        if collisions:
            review_id = NotesInteropService._ensure_folder_placement_review(
                cursor,
                profile=server_profile_id,
                dataset=dataset_id,
                requested_name=str(receipt["requested_folder_name"]),
                collision_ids=collisions,
            )
            now = _utc_now()
            cursor.execute(
                "UPDATE note_organization_receipts SET state = 'placement_review', "
                "review_id = ?, collision_ids_json = ?, updated_at = ? "
                "WHERE receipt_id = ? AND state = 'pending_organization'",
                (
                    review_id,
                    json.dumps(list(collisions), separators=(",", ":")),
                    now,
                    str(receipt["receipt_id"]),
                ),
            )
            organization_version = db._library_organization_for_notes(
                cursor, [str(note["id"])]
            )[str(note["id"])]["organization_version"]
            cursor.execute(
                "UPDATE note_organization_receipts SET organization_version = ? "
                "WHERE receipt_id = ? AND state = 'placement_review'",
                (organization_version, str(receipt["receipt_id"])),
            )
            counts["placement_review"] += 1
        else:
            cursor.execute(
                "DELETE FROM note_organization_receipts WHERE receipt_id = ? "
                "AND state = 'pending_organization'",
                (str(receipt["receipt_id"]),),
            )
            counts["finalized"] += 1
        if self.failure_injector is not None:
            self.failure_injector("after_receipt_finalization_intents")

    @staticmethod
    def _insert_note_sync_intent(
        cursor: Any,
        note: Any,
        *,
        intent_id: str,
        server_profile_id: str,
        dataset_id: str,
        base_version: int | None,
        request_fingerprint: str,
    ) -> None:
        operation = "create" if int(note["version"]) == 1 else "update"
        payload = {
            "id": str(note["id"]),
            "title": str(note["title"]),
            "content": str(note["content"]),
            "created_at": str(note["created_at"]),
            "last_modified": str(note["last_modified"]),
            "deleted": int(note["deleted"]),
            "client_id": str(note["client_id"]),
            "version": int(note["version"]),
        }
        cursor.execute(
            "INSERT INTO note_sync_publication_intents("
            "intent_id, server_profile_id, dataset_id, note_id, operation, "
            "base_version, entity_version, request_fingerprint, payload_json, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                intent_id,
                server_profile_id,
                dataset_id,
                str(note["id"]),
                operation,
                base_version,
                int(note["version"]),
                request_fingerprint,
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                _utc_now(),
            ),
        )

    def _finalize_resolved_placement_receipt(
        self,
        cursor: Any,
        *,
        receipt: Any,
        note: Any,
        server_profile_id: str,
        dataset_id: str,
    ) -> bool:
        review = cursor.execute(
            "SELECT resolution, local_object_id FROM notes_organization_adoption_reviews "
            "WHERE review_id = ? AND state = 'resolved'",
            (str(receipt["review_id"]),),
        ).fetchone()
        if review is None:
            return False
        resolution = str(review["resolution"])
        if resolution != "keep_local":
            if resolution == "merge":
                folder = cursor.execute(
                    "SELECT * FROM note_folders WHERE id = ? AND deleted = 0",
                    (str(review["local_object_id"]),),
                ).fetchone()
            else:
                renamed = cursor.execute(
                    "SELECT folder.*, parent.sync_id AS parent_sync_id "
                    "FROM note_folders AS folder LEFT JOIN note_folders AS parent "
                    "ON parent.id = folder.parent_id WHERE folder.id = ? "
                    "AND folder.deleted = 0",
                    (str(review["local_object_id"]),),
                ).fetchone()
                if renamed is not None and renamed["sync_id"]:
                    NotesInteropService._record_organization_intent(
                        NotesOrganizationRepository(
                            self.notes_repository.db,
                            server_profile_id=server_profile_id,
                        ),
                        cursor,
                        profile=server_profile_id,
                        dataset=dataset_id,
                        domain="notes.folder",
                        object_id=str(renamed["sync_id"]),
                        payload={
                            "name": str(renamed["name"]),
                            "parent_sync_id": (
                                str(renamed["parent_sync_id"])
                                if renamed["parent_sync_id"] is not None
                                else None
                            ),
                        },
                        source_version=int(renamed["version"]),
                    )
                folder, created, collisions = NotesInteropService._ensure_save_folder(
                    cursor,
                    folder=str(receipt["requested_folder_name"]),
                    folder_sync_id=None,
                )
                if collisions:
                    return False
                if created and folder is not None:
                    NotesInteropService._record_organization_intent(
                        NotesOrganizationRepository(
                            self.notes_repository.db,
                            server_profile_id=server_profile_id,
                        ),
                        cursor,
                        profile=server_profile_id,
                        dataset=dataset_id,
                        domain="notes.folder",
                        object_id=str(folder["sync_id"]),
                        payload={"name": str(folder["name"]), "parent_sync_id": None},
                        source_version=int(folder["version"]),
                    )
            if folder is None or not folder["sync_id"]:
                return False
            LocalNoteFolderRepository(self.notes_repository.db).attach_manual(
                folder_id=str(folder["id"]), note_id=str(note["id"]), cursor=cursor
            )
            NotesInteropService._record_organization_link_intent(
                NotesOrganizationRepository(
                    self.notes_repository.db, server_profile_id=server_profile_id
                ),
                cursor,
                profile=server_profile_id,
                dataset=dataset_id,
                domain="notes.folder_link",
                members=(str(note["id"]), str(folder["sync_id"])),
                payload={
                    "note_id": str(note["id"]),
                    "folder_sync_id": str(folder["sync_id"]),
                },
            )
        cursor.execute(
            "DELETE FROM note_organization_receipts WHERE receipt_id = ?",
            (str(receipt["receipt_id"]),),
        )
        return True

    def _cancel_receipt_with_cursor(self, cursor: Any, receipt: Any) -> None:
        self.notes_repository.db._cancel_note_organization_receipt_with_cursor(
            cursor, receipt
        )

    def drain_pending_note_intents(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> dict[str, int]:
        """Project each latest dispatchable note sync intent idempotently."""

        if self.notes_producer is None:
            return {"copied": 0, "already_copied": 0}
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        dataset_id = str(profile.get("dataset_id") or "") if profile else ""
        self.finalize_pending_note_organization_receipts(
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
        )
        existing_ids = {
            item["client_envelope_id"]
            for item in self.state_repository.list_sync_v2_outbox_entries(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=dataset_id,
            )
        }
        copied = 0
        already = 0
        for row in self.notes_repository.db.list_latest_dispatchable_note_sync_entries(
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
        ):
            payload = json.loads(str(row["payload"]))
            note_id = str(row["entity_id"])
            if str(row["operation"]) == "delete":
                result = self.notes_producer.enqueue_note_delete(
                    server_profile_id=server_profile_id,
                    authenticated_principal_id=authenticated_principal_id,
                    workspace_scope=workspace_scope,
                    note_id=note_id,
                    base_version=row["base_version"],
                    entity_version=int(row["version"]),
                    publication_intent_id=f"notes-publication:{row['intent_id']}",
                )
            else:
                result = self.notes_producer.enqueue_note_upsert(
                    server_profile_id=server_profile_id,
                    authenticated_principal_id=authenticated_principal_id,
                    workspace_scope=workspace_scope,
                    note_id=note_id,
                    title=str(payload.get("title") or ""),
                    content=str(payload.get("content") or ""),
                    status="active",
                    base_version=row["base_version"],
                    entity_version=int(row["version"]),
                    publication_intent_id=f"notes-publication:{row['intent_id']}",
                )
            if result.get("status") != "enqueued":
                continue
            after_id = result["outbox_entry"]["client_envelope_id"]
            if after_id in existing_ids:
                already += 1
            else:
                copied += 1
                existing_ids.add(after_id)
            with self.notes_repository.db.transaction() as cursor:
                cursor.execute(
                    "UPDATE note_sync_publication_intents SET "
                    "outbox_client_envelope_id = ?, copied_at = COALESCE(copied_at, ?) "
                    "WHERE intent_id = ? AND acknowledged_at IS NULL",
                    (after_id, _utc_now(), str(row["intent_id"])),
                )
        return {"copied": copied, "already_copied": already}

    def _record_server_enrollment_status(
        self,
        *,
        server_profile_id: str,
        dataset_id: str,
        bootstrap_id: str,
        state: str,
        captured_count: int,
        expected_count: int,
        error_code: str | None,
    ) -> Any:
        if state not in {"initializing", "ready", "failed"}:
            raise ValueError("invalid Notes organization server state")
        local_state = (
            "failed"
            if state == "failed"
            else ("pulling" if state == "ready" else "initializing")
        )
        now = datetime.now(UTC).isoformat()
        with self.notes_repository.db.transaction() as cursor:
            existing = cursor.execute(
                "SELECT * FROM notes_organization_sync_checkpoints WHERE "
                "server_profile_id = ? AND dataset_id = ?",
                (server_profile_id, dataset_id),
            ).fetchone()
            if existing is None:
                cursor.execute(
                    "INSERT INTO notes_organization_sync_checkpoints("
                    "server_profile_id, dataset_id, local_state, server_state, "
                    "bootstrap_id, captured_count, expected_count, error_code, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        server_profile_id,
                        dataset_id,
                        local_state,
                        state,
                        bootstrap_id,
                        captured_count,
                        expected_count,
                        error_code,
                        now,
                    ),
                )
            else:
                cursor.execute(
                    "UPDATE notes_organization_sync_checkpoints SET local_state = ?, "
                    "server_state = ?, "
                    "captured_count = ?, expected_count = ?, error_code = ?, updated_at = ? "
                    "WHERE server_profile_id = ? AND dataset_id = ?",
                    (
                        local_state if state != "ready" else existing["local_state"],
                        state,
                        captured_count,
                        expected_count,
                        error_code,
                        now,
                        server_profile_id,
                        dataset_id,
                    ),
                )
            return cursor.execute(
                "SELECT * FROM notes_organization_sync_checkpoints WHERE "
                "server_profile_id = ? AND dataset_id = ?",
                (server_profile_id, dataset_id),
            ).fetchone()

    def _set_enrollment_local_state(
        self,
        *,
        server_profile_id: str,
        dataset_id: str,
        local_state: str,
        pull_cursor: Any,
        error_code: str | None,
    ) -> None:
        with self.notes_repository.db.transaction() as cursor:
            cursor.execute(
                "UPDATE notes_organization_sync_checkpoints SET local_state = ?, "
                "pull_cursor = COALESCE(?, pull_cursor), error_code = ?, updated_at = ? "
                "WHERE server_profile_id = ? AND dataset_id = ?",
                (
                    local_state,
                    None if pull_cursor is None else str(pull_cursor),
                    error_code,
                    datetime.now(UTC).isoformat(),
                    server_profile_id,
                    dataset_id,
                ),
            )

    @staticmethod
    def _dataset_id(profile: dict[str, Any]) -> str:
        dataset_id = str(profile.get("dataset_id") or "")
        if not dataset_id:
            raise ValueError("synchronized profile is missing dataset identity")
        return dataset_id

    @staticmethod
    def _required_sync_id(row: Any) -> str:
        sync_id = row["sync_id"]
        if not sync_id:
            raise ValueError("organization resource is not synchronized")
        return str(sync_id)

    @staticmethod
    def _resource_sync_id(cursor: Any, table: str, item_id: int) -> str:
        if table not in _RESOURCE_SYNC_ID_TABLES:
            raise ValueError("unsupported organization resource table")
        row = cursor.execute(
            f"SELECT sync_id FROM {table} WHERE id = ?", (item_id,)
        ).fetchone()
        if row is None:
            raise ValueError("organization resource does not exist")
        if row["sync_id"]:
            return str(row["sync_id"])
        sync_id = new_organization_sync_id()
        cursor.execute(
            f"UPDATE {table} SET sync_id = ? WHERE id = ? AND sync_id IS NULL",
            (sync_id, item_id),
        )
        return sync_id

    @staticmethod
    def _folder_link_effective(cursor: Any, folder_id: str, note_id: str) -> bool:
        return (
            cursor.execute(
                "SELECT 1 FROM note_folder_memberships m "
                "JOIN note_folders f ON f.id = m.folder_id "
                "WHERE m.folder_id = ? AND m.note_id = ? AND m.deleted = 0 "
                "AND (m.ownership = 'manual' OR m.owner_active = 1) "
                "AND NOT EXISTS (SELECT 1 FROM note_folder_sync_suppressions s "
                "WHERE s.note_id = m.note_id AND s.folder_sync_id = f.sync_id) LIMIT 1",
                (folder_id, note_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _effective_folder_links(cursor: Any) -> set[tuple[str, str]]:
        return {
            (str(row["folder_id"]), str(row["note_id"]))
            for row in cursor.execute(
                "SELECT DISTINCT m.folder_id, m.note_id FROM note_folder_memberships m "
                "JOIN note_folders f ON f.id = m.folder_id "
                "WHERE m.deleted = 0 AND (m.ownership = 'manual' OR m.owner_active = 1) "
                "AND NOT EXISTS (SELECT 1 FROM note_folder_sync_suppressions s "
                "WHERE s.note_id = m.note_id AND s.folder_sync_id = f.sync_id)"
            ).fetchall()
        }

    def _record_resource(
        self,
        cursor: Any,
        *,
        domain: str,
        object_id: str,
        operation: str,
        payload: dict[str, object],
        source_version: int | None,
        profile: str,
        dataset: str,
        routing_metadata: dict[str, object] | None = None,
    ) -> str | None:
        if self._keep_local_blocks_intent(
            cursor,
            profile=profile,
            dataset=dataset,
            domain=domain,
            object_id=object_id,
            payload=payload,
        ):
            return None
        if source_version is None:
            return self.notes_repository._record_inferred_intent_with_cursor(
                cursor,
                profile=profile,
                dataset=dataset,
                domain=domain,
                object_id=object_id,
                operation=operation,
                payload=payload,
            )
        return self.notes_repository.record_intent(
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

    def _record_link(
        self,
        cursor: Any,
        *,
        domain: str,
        object_id: str,
        operation: str,
        payload: dict[str, object],
        profile: str,
        dataset: str,
    ) -> str | None:
        return self._record_resource(
            cursor,
            domain=domain,
            object_id=object_id,
            operation=operation,
            payload=payload,
            source_version=None,
            profile=profile,
            dataset=dataset,
        )

    def _record_folder_link(
        self,
        cursor: Any,
        *,
        folder_id: str,
        note_id: str,
        operation: str,
        profile: str,
        dataset: str,
    ) -> str | None:
        folder = cursor.execute(
            "SELECT sync_id FROM note_folders WHERE id = ?", (folder_id,)
        ).fetchone()
        if folder is None or not folder["sync_id"]:
            raise ValueError("folder is not synchronized")
        payload: dict[str, object] = {
            "note_id": note_id,
            "folder_sync_id": str(folder["sync_id"]),
        }
        object_id = organization_link_id(
            "notes.folder_link", (note_id, str(folder["sync_id"]))
        )
        return self._record_link(
            cursor,
            domain="notes.folder_link",
            object_id=object_id,
            operation=operation,
            payload=payload,
            profile=profile,
            dataset=dataset,
        )

    def _record_keyword_link(
        self,
        cursor: Any,
        *,
        subject_type: str,
        subject_id: str,
        keyword_sync_id: str,
        operation: str,
        profile: str,
        dataset: str,
    ) -> str | None:
        payload: dict[str, object] = {
            "subject_type": subject_type,
            "subject_id": subject_id,
            "keyword_sync_id": keyword_sync_id,
        }
        object_id = organization_link_id(
            "notes.keyword_link", (subject_type, subject_id, keyword_sync_id)
        )
        return self._record_link(
            cursor,
            domain="notes.keyword_link",
            object_id=object_id,
            operation=operation,
            payload=payload,
            profile=profile,
            dataset=dataset,
        )

    def rename_folder(
        self,
        *,
        folder_repository: Any,
        folder_id: str,
        name: str,
        expected_version: int,
        **scope: Any,
    ) -> Any:
        return self._mutate_folder(
            folder_repository=folder_repository,
            method="rename_folder",
            folder_id=folder_id,
            operation="upsert",
            method_arguments={"name": name, "expected_version": expected_version},
            **scope,
        )

    def move_folder(
        self,
        *,
        folder_repository: Any,
        folder_id: str,
        parent_id: str | None,
        expected_version: int,
        **scope: Any,
    ) -> Any:
        return self._mutate_folder(
            folder_repository=folder_repository,
            method="move_folder",
            folder_id=folder_id,
            operation="upsert",
            method_arguments={
                "parent_id": parent_id,
                "expected_version": expected_version,
            },
            **scope,
        )

    def delete_folder(
        self,
        *,
        folder_repository: Any,
        folder_id: str,
        expected_version: int,
        **scope: Any,
    ) -> Any:
        return self._mutate_folder(
            folder_repository=folder_repository,
            method="soft_delete_folder",
            folder_id=folder_id,
            operation="tombstone",
            method_arguments={"expected_version": expected_version},
            **scope,
        )

    def restore_folder(
        self,
        *,
        folder_repository: Any,
        folder_id: str,
        expected_version: int,
        **scope: Any,
    ) -> Any:
        return self._mutate_folder(
            folder_repository=folder_repository,
            method="restore_folder",
            folder_id=folder_id,
            operation="upsert",
            method_arguments={"expected_version": expected_version},
            **scope,
        )

    def _mutate_folder(
        self,
        *,
        folder_repository: Any,
        method: str,
        folder_id: str,
        operation: str,
        method_arguments: dict[str, Any],
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
    ) -> Any:
        profile = self.state_repository.get_sync_v2_profile_state(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
        )
        mutate = getattr(folder_repository, method)
        if profile is None:
            raise ValueError("persisted Notes profile scope is required")
        if profile.get("profile_mode") == "local_only":
            return mutate(folder_id, **method_arguments)
        dataset_id = str(profile.get("dataset_id") or "")
        if not dataset_id:
            raise ValueError("synchronized profile is missing dataset identity")
        with self.notes_repository.db.transaction() as cursor:
            self._require_ready(
                cursor,
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
            )
            before = cursor.execute(
                "SELECT sync_id FROM note_folders WHERE id = ?", (folder_id,)
            ).fetchone()
            if before is None or not before["sync_id"]:
                raise ValueError("folder is not synchronized")
            result = mutate(folder_id, cursor=cursor, **method_arguments)
            payload: dict[str, object] = {}
            if operation == "upsert":
                parent_sync_id = None
                if result.folder.parent_id is not None:
                    parent = cursor.execute(
                        "SELECT sync_id FROM note_folders WHERE id = ?",
                        (result.folder.parent_id,),
                    ).fetchone()
                    if parent is None or not parent["sync_id"]:
                        raise ValueError("parent folder is not synchronized")
                    parent_sync_id = str(parent["sync_id"])
                payload = {
                    "name": result.folder.name,
                    "parent_sync_id": parent_sync_id,
                }
            self._record_resource(
                cursor,
                profile=server_profile_id,
                dataset=dataset_id,
                domain="notes.folder",
                object_id=str(before["sync_id"]),
                operation=operation,
                payload=payload,
                source_version=result.folder.version,
                routing_metadata=(
                    {"restore_intent": True} if method == "restore_folder" else None
                ),
            )
        return result

    @staticmethod
    def _require_ready(cursor: Any, *, server_profile_id: str, dataset_id: str) -> None:
        row = cursor.execute(
            """
            SELECT local_state, server_state, inventory_phase, error_code
              FROM notes_organization_sync_checkpoints
             WHERE server_profile_id = ? AND dataset_id = ?
            """,
            (server_profile_id, dataset_id),
        ).fetchone()
        if (
            row is None
            or row["local_state"] != "ready"
            or row["server_state"] != "ready"
            or row["inventory_phase"] != "complete"
            or row["error_code"] is not None
        ):
            raise ValueError("Notes organization group is not ready")

    def drain_pending_intents(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        device_id: str,
    ) -> dict[str, int]:
        """Insert-or-confirm every unacknowledged intent, then mark it copied."""

        self.finalize_pending_note_organization_receipts(
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
        )
        connection = self.notes_repository.db.get_connection()
        self._require_ready(
            connection,
            server_profile_id=server_profile_id,
            dataset_id=dataset_id,
        )
        rows = connection.execute(
            """
            SELECT * FROM notes_organization_sync_intents
             WHERE server_profile_id = ? AND dataset_id = ?
               AND acknowledged_at IS NULL
             ORDER BY intent_sequence
            """,
            (server_profile_id, dataset_id),
        ).fetchall()
        copied = 0
        already_copied = 0
        for row in rows:
            row = self._bind_predecessor_base_if_ready(
                row,
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
            )
            if row is None:
                continue
            intent_id = str(row["intent_id"])
            existing = self.state_repository.list_sync_v2_outbox_entries(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=dataset_id,
                client_envelope_ids=[intent_id],
            )
            envelope = self._envelope(row, device_id=device_id)
            payload = envelope.get("payload") or envelope.get("payload_clear") or {}
            if self._keep_local_blocks_intent(
                connection,
                profile=server_profile_id,
                dataset=dataset_id,
                domain=str(row["domain"]),
                object_id=str(row["object_id"]),
                payload=payload,
            ):
                continue
            if self.failure_injector is not None:
                self.failure_injector("before_outbox_insert")
            self.state_repository.enqueue_sync_v2_outbox_envelope(
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                dataset_id=dataset_id,
                envelope=envelope,
            )
            if self.failure_injector is not None:
                self.failure_injector("after_outbox_insert")
            with self.notes_repository.db.transaction() as cursor:
                cursor.execute(
                    """
                    UPDATE notes_organization_sync_intents
                       SET outbox_client_envelope_id = ?,
                           copied_at = COALESCE(copied_at, ?)
                     WHERE intent_id = ? AND acknowledged_at IS NULL
                    """,
                    (intent_id, _utc_now(), intent_id),
                )
            if existing:
                already_copied += 1
            else:
                copied += 1
        return {"copied": copied, "already_copied": already_copied}

    def _bind_predecessor_base_if_ready(
        self,
        row: Any,
        *,
        server_profile_id: str,
        dataset_id: str,
    ) -> Any | None:
        """Bind a successor's accepted predecessor head exactly once."""

        predecessor_intent_id = row["predecessor_intent_id"]
        if predecessor_intent_id is None:
            return row
        base = (
            row["base_server_cursor"],
            row["base_object_revision"],
            row["base_object_hash"],
        )
        if all(value is not None for value in base):
            return row

        with self.notes_repository.db.transaction() as cursor:
            current = cursor.execute(
                "SELECT * FROM notes_organization_sync_intents "
                "WHERE server_profile_id = ? AND dataset_id = ? AND intent_id = ? "
                "AND acknowledged_at IS NULL",
                (server_profile_id, dataset_id, str(row["intent_id"])),
            ).fetchone()
            if current is None:
                return None
            predecessor = cursor.execute(
                "SELECT domain, object_id, payload_hash, acknowledged_at "
                "FROM notes_organization_sync_intents "
                "WHERE server_profile_id = ? AND dataset_id = ? AND intent_id = ?",
                (
                    server_profile_id,
                    dataset_id,
                    str(predecessor_intent_id),
                ),
            ).fetchone()
            if predecessor is None or predecessor["acknowledged_at"] is None:
                return None
            head = cursor.execute(
                "SELECT server_cursor, object_revision, object_hash, apply_state "
                "FROM notes_organization_heads WHERE server_profile_id = ? "
                "AND dataset_id = ? AND domain = ? AND object_id = ?",
                (
                    server_profile_id,
                    dataset_id,
                    str(predecessor["domain"]),
                    str(predecessor["object_id"]),
                ),
            ).fetchone()
            if (
                head is None
                or str(head["apply_state"]) != "applied"
                or str(head["object_hash"]) != str(predecessor["payload_hash"])
            ):
                return None
            cursor.execute(
                "UPDATE notes_organization_sync_intents "
                "SET base_server_cursor = ?, base_object_revision = ?, "
                "base_object_hash = ? WHERE intent_id = ? "
                "AND base_server_cursor IS NULL AND base_object_revision IS NULL "
                "AND base_object_hash IS NULL",
                (
                    str(head["server_cursor"]),
                    int(head["object_revision"]),
                    str(head["object_hash"]),
                    str(current["intent_id"]),
                ),
            )
            return cursor.execute(
                "SELECT * FROM notes_organization_sync_intents WHERE intent_id = ?",
                (str(current["intent_id"]),),
            ).fetchone()

    @staticmethod
    def _keep_local_blocks_intent(
        cursor: Any,
        *,
        profile: str,
        dataset: str,
        domain: str,
        object_id: str,
        payload: Any,
    ) -> bool:
        """Keep explicitly local identities and all dependent links unpublished."""

        reviews = cursor.execute(
            "SELECT domain, local_object_id, remote_object_id FROM "
            "notes_organization_adoption_reviews WHERE server_profile_id = ? "
            "AND dataset_id = ? AND state = 'resolved' AND resolution = 'keep_local'",
            (profile, dataset),
        ).fetchall()
        blocked: set[str] = set()
        tables = {
            "notes.keyword": "keywords",
            "notes.keyword_collection": "keyword_collections",
            "notes.folder": "note_folders",
        }
        for review in reviews:
            blocked.update(
                {
                    str(review["local_object_id"]),
                    str(review["remote_object_id"]),
                }
            )
            table = tables.get(str(review["domain"]))
            if table is None:
                continue
            resource = cursor.execute(
                f"SELECT sync_id FROM {table} WHERE id = ?",
                (review["local_object_id"],),
            ).fetchone()
            if resource is not None and resource["sync_id"]:
                blocked.add(str(resource["sync_id"]))
        if object_id in blocked:
            return True
        if not isinstance(payload, dict):
            return False
        return any(str(value) in blocked for value in payload.values())

    def reconcile_acknowledgements(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
    ) -> int:
        """Mark local intents acknowledged after their general row is dispatched."""

        rows = self.state_repository.list_sync_v2_outbox_entries(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=dataset_id,
            status="dispatched",
        )
        accepted_organization_rows = [
            row
            for row in rows
            if row["domain"] in NOTES_ORGANIZATION_DOMAINS
            and isinstance(row.get("accepted_result"), dict)
            and row["accepted_result"].get("apply_status") == "applied"
            and row["accepted_result"].get("server_cursor") is not None
            and row["accepted_result"].get("object_revision") is not None
        ]
        accepted_note_rows = [
            row
            for row in rows
            if row["domain"] == "notes"
            and isinstance(row.get("accepted_result"), dict)
            and row["accepted_result"].get("apply_status") == "applied"
        ]
        if not accepted_organization_rows and not accepted_note_rows:
            return 0
        if self.failure_injector is not None:
            self.failure_injector("after_server_acknowledgement")
        acknowledged = 0
        now = _utc_now()
        with self.notes_repository.db.transaction() as cursor:
            for row in accepted_note_rows:
                result = cursor.execute(
                    "UPDATE note_sync_publication_intents SET "
                    "acknowledged_at = COALESCE(acknowledged_at, ?) "
                    "WHERE server_profile_id = ? AND dataset_id = ? "
                    "AND outbox_client_envelope_id = ? AND acknowledged_at IS NULL",
                    (
                        now,
                        server_profile_id,
                        dataset_id,
                        str(row["client_envelope_id"]),
                    ),
                )
                acknowledged += int(result.rowcount)
            for row in accepted_organization_rows:
                accepted = row["accepted_result"]
                intent = cursor.execute(
                    "SELECT * FROM notes_organization_sync_intents "
                    "WHERE server_profile_id = ? AND dataset_id = ? "
                    "AND intent_id = ? AND acknowledged_at IS NULL",
                    (
                        server_profile_id,
                        dataset_id,
                        str(row["client_envelope_id"]),
                    ),
                ).fetchone()
                if intent is None:
                    continue
                cursor.execute(
                    """
                    INSERT INTO notes_organization_heads(
                        server_profile_id, dataset_id, domain, object_id,
                        operation, schema_version, encryption_policy,
                        payload_json, payload_hash, object_revision, object_hash,
                        server_cursor, deleted, apply_state, applied_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, 'server_trusted_v1', ?, ?, ?, ?, ?, ?,
                              'applied', ?, ?)
                    ON CONFLICT(server_profile_id, dataset_id, domain, object_id)
                    DO UPDATE SET
                        operation = excluded.operation,
                        payload_json = excluded.payload_json,
                        payload_hash = excluded.payload_hash,
                        object_revision = excluded.object_revision,
                        object_hash = excluded.object_hash,
                        server_cursor = excluded.server_cursor,
                        deleted = excluded.deleted,
                        apply_state = 'applied',
                        applied_at = excluded.applied_at,
                        updated_at = excluded.updated_at
                    WHERE CAST(notes_organization_heads.server_cursor AS INTEGER)
                          <= CAST(excluded.server_cursor AS INTEGER)
                    """,
                    (
                        server_profile_id,
                        dataset_id,
                        str(intent["domain"]),
                        str(intent["object_id"]),
                        str(intent["operation"]),
                        str(intent["payload_json"]),
                        str(intent["payload_hash"]),
                        int(accepted["object_revision"]),
                        str(intent["payload_hash"]),
                        str(accepted["server_cursor"]),
                        int(str(intent["operation"]) == "tombstone"),
                        now,
                        now,
                    ),
                )
                result = cursor.execute(
                    "UPDATE notes_organization_sync_intents "
                    "SET acknowledged_at = COALESCE(acknowledged_at, ?) "
                    "WHERE intent_id = ? AND acknowledged_at IS NULL",
                    (now, str(intent["intent_id"])),
                )
                acknowledged += int(result.rowcount)
        return acknowledged

    @staticmethod
    def _envelope(row: Any, *, device_id: str) -> dict[str, Any]:
        envelope = NotesSyncV2OutboxProducer.build_organization_envelope(
            row, device_id=device_id
        )
        envelope["routing_metadata"] = json.loads(str(row["routing_metadata_json"]))
        envelope["base_server_cursor"] = row["base_server_cursor"]
        return envelope


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
