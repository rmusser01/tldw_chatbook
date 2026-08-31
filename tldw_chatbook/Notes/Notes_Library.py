# Notes_Library.py
# Description: This module provides a service layer for managing notes and note keywords
#
from __future__ import annotations

# Imports
import hashlib
import logging
import re
import threading
import sqlite3  # For exception handling in _get_db
import time
import json
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, List, Dict, Optional, Any, Sequence, Union

#
# Third-Party Imports
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    SchemaError,
)
from tldw_chatbook.config import (
    chachanotes_db as global_db_from_config,
    load_console_library_migration_seed,
)
from tldw_chatbook.Utils.private_paths import (
    lexical_path,
    verify_trusted_directory,
)
from tldw_chatbook.Notes.note_folder_models import (
    NotesOrganizationRepositoryError,
    portable_collision_key,
    portable_relative_path,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from ..Metrics.metrics_logger import log_counter, log_histogram

if TYPE_CHECKING:
    from tldw_chatbook.Notes.notes_organization_repository import (
        NotesOrganizationRepository,
    )
#
#######################################################################################################################
#
# Functions:

logger = logging.getLogger(__name__)

_RESEARCH_RECEIPT_PROOF_PREFIX = "research-receipt-proof:"


def _is_internal_research_keyword(row: Any) -> bool:
    return isinstance(row, dict) and re.fullmatch(
        rf"{re.escape(_RESEARCH_RECEIPT_PROOF_PREFIX)}[0-9a-f]{{64}}",
        str(row.get("keyword") or ""),
    ) is not None


class NotesInteropService:
    def __init__(
        self,
        base_db_directory: Union[str, Path],
        api_client_id: str,  # This api_client_id might be a fallback or general app id
        global_db_to_use: Optional[CharactersRAGDB] = None,
        failure_injector: Optional[Callable[[str], None]] = None,
    ):

        self.base_db_directory = lexical_path(base_db_directory)
        # self.api_client_id is not directly used if _get_db uses user_id as client_id
        # It's good to have it for context or if some methods need a generic app client_id.
        self.api_client_id = api_client_id
        self._organization_failure_injector = failure_injector

        self._db_instances: Dict[
            str, CharactersRAGDB
        ] = {}  # Cache instances per user_id (as client_id)
        self._db_lock = threading.Lock()

        try:
            verify_trusted_directory(
                self.base_db_directory,
                allow_shared_sticky=False,
            )
            logger.info(
                f"NotesInteropService: Verified base directory: {self.base_db_directory}"
            )
        except OSError as e:
            logger.error(
                f"Failed to verify base DB directory {self.base_db_directory}: {e}"
            )
            raise CharactersRAGDBError(
                f"Failed to verify base DB directory {self.base_db_directory}: {e}"
            ) from e

        # Store the global DB instance
        if global_db_to_use:
            self.unified_db_template = global_db_to_use  # Store the template instance
            logger.info(
                f"NotesInteropService initialized with unified DB template: {self.unified_db_template.db_path_str}"
            )
        elif global_db_from_config:
            self.unified_db_template = global_db_from_config
            logger.info(
                f"NotesInteropService using imported global config DB template: {self.unified_db_template.db_path_str}"
            )
        else:
            self.unified_db_template = None

        if not self.unified_db_template:
            logger.critical(
                "NotesInteropService CRITICAL: No unified database template instance available!"
            )
            raise CharactersRAGDBError(
                "No unified database template for NotesInteropService."
            )

    def _get_db(self, user_id: str) -> CharactersRAGDB:
        """
        Retrieves or creates a CharactersRAGDB instance for a given user_id,
        always pointing to the single, unified database file.
        The `user_id` is used as the `client_id` for the returned DB instance.
        Instances are cached per `user_id`. This method is thread-safe.

        Args:
            user_id: The unique identifier for the user, used as client_id for DB operations.

        Returns:
            A CharactersRAGDB instance configured for the specified user_id
            but operating on the global database file.

        Raises:
            ValueError: If user_id is empty or invalid.
            CharactersRAGDBError: If the unified database template is not available or
                                  if database initialization for the user context fails.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string for DB operations.")
        user_id = user_id.strip()

        # Fast path: check if instance already exists for this user_id (as client_id)
        if user_id in self._db_instances:
            # The cached instance already uses user_id as its client_id
            # and points to the correct global DB file.
            return self._db_instances[user_id]

        # Slow path: acquire lock and double-check cache
        with self._db_lock:
            if user_id in self._db_instances:  # Double-check
                return self._db_instances[user_id]

            if not self.unified_db_template:
                logger.critical(
                    "NotesInteropService: Unified database template (self.unified_db_template) is not initialized!"
                )
                raise CharactersRAGDBError(
                    "Unified database template is not available in NotesInteropService."
                )

            # Create a new CharactersRAGDB instance for this user_id,
            # ensuring it points to the *global unified database file path*
            # and uses the current `user_id` as its `client_id`.
            try:
                unified_db_file_path = self.unified_db_template.db_path_str
                logger.info(
                    f"Creating new CharactersRAGDB instance for context of user '{user_id}'. "
                    f"DB File: {unified_db_file_path}, Client ID for ops: '{user_id}'."
                )

                db_instance = CharactersRAGDB(
                    db_path=unified_db_file_path,  # Use the path from the unified DB template
                    client_id=user_id,  # Use the passed user_id as the client_id for this instance
                    console_library_migration_seed=load_console_library_migration_seed(),
                )
                self._db_instances[user_id] = db_instance  # Cache it
                logger.info(
                    f"Successfully initialized dynamic CharactersRAGDB instance for user context '{user_id}'."
                )
                return db_instance
            except (CharactersRAGDBError, SchemaError, sqlite3.Error) as e:
                logger.error(
                    f"Failed to initialize dynamic CharactersRAGDB instance for user '{user_id}': {e}",
                    exc_info=True,
                )
                raise
            except Exception as e:  # Catch any other unexpected Python error
                logger.error(
                    f"Unexpected error initializing dynamic CharactersRAGDB for user '{user_id}': {e}",
                    exc_info=True,
                )
                raise CharactersRAGDBError(
                    f"Unexpected error initializing DB instance for user {user_id}: {e}"
                ) from e

    # --- Note Methods ---

    def note_transaction(self, user_id: str) -> Any:
        """Return the canonical Notes transaction for one mutation owner."""

        return self._get_db(user_id).transaction(immediate=True)

    def notes_db(self, user_id: str) -> CharactersRAGDB:
        """Return the exact database owner used for one user's Notes mutation."""

        return self._get_db(user_id)

    def add_internal_research_quick_note_owner_proof(
        self, user_id: str, note_id: str, owner_proof: str
    ) -> bool:
        """Store one private recovery proof outside all ordinary Notes metadata."""

        return self._get_db(user_id).add_research_quick_note_owner_proof(
            note_id, owner_proof
        )

    def has_internal_research_quick_note_owner_proof(
        self, user_id: str, note_id: str, owner_proof: str
    ) -> bool:
        """Verify one exact private recovery proof without returning its payload."""

        return self._get_db(user_id).has_research_quick_note_owner_proof(
            note_id, owner_proof
        )

    def remove_internal_research_quick_note_owner_proof(
        self, user_id: str, note_id: str, owner_proof: str
    ) -> bool:
        """Remove only the exact private proof held by the caller."""

        return self._get_db(user_id).remove_research_quick_note_owner_proof(
            note_id, owner_proof
        )

    def add_note(
        self, user_id: str, title: str, content: str, note_id: Optional[str] = None
    ) -> str:
        """
        Adds a new note for the specified user. The user_id will be used as the client_id.
        """
        start_time = time.time()
        log_counter(
            "notes_library_add_note_attempt",
            labels={"has_note_id": str(note_id is not None)},
        )

        try:
            db = self._get_db(user_id)
            created_note_id = db.add_note(title=title, content=content, note_id=note_id)
            if created_note_id is None:
                logger.error(
                    f"add_note for user_id '{user_id}' (as client_id) returned None unexpectedly for title '{title}'."
                )
                log_counter(
                    "notes_library_add_note_error",
                    labels={"error_type": "null_id_returned"},
                )
                raise CharactersRAGDBError(
                    "Failed to create note, received None ID unexpectedly."
                )

            # Log success metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_add_note_duration",
                duration,
                labels={"status": "success"},
            )
            log_histogram("notes_library_note_content_length", len(content))
            log_counter("notes_library_add_note_success")

            return created_note_id
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_add_note_duration", duration, labels={"status": "error"}
            )
            log_counter(
                "notes_library_add_note_error", labels={"error_type": type(e).__name__}
            )
            raise

    def get_note_by_id(self, user_id: str, note_id: str) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        log_counter("notes_library_get_note_attempt")

        db = self._get_db(
            user_id
        )  # user_id here is mainly for consistency or if _get_db has other uses
        result = db.get_note_by_id(
            note_id=note_id
        )  # The actual filtering by user would be in SQL if notes were user-specific

        # Log metrics
        duration = time.time() - start_time
        log_histogram("notes_library_get_note_duration", duration)
        log_counter(
            "notes_library_get_note_result", labels={"found": str(result is not None)}
        )

        return result

    def get_note_version_states(
        self, user_id: str, note_ids: Sequence[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Read only (version, deleted) for the given note ids (TASK-23027).

        One consistent snapshot for the lasting-sync observer's change check;
        see ``CharactersRAGDB.get_note_version_states`` for the contract.
        """

        db = self._get_db(user_id)
        return db.get_note_version_states(note_ids)

    def get_agent_lesson_preflight_snapshot(
        self, user_id: str, note_id: str
    ) -> Dict[str, Any]:
        """Return one private, transaction-consistent lesson review snapshot.

        Unlike the public Library projection, this in-process seam returns all
        exact active keyword names and the actual unresolved receipt state.
        It contains no note title or body and is not registered as a tool.
        """

        db = self._get_db(user_id)
        with db.transaction() as cursor:
            note = cursor.execute(
                "SELECT version FROM notes WHERE id = ? AND deleted = 0",
                (note_id,),
            ).fetchone()
            if note is None:
                raise ValueError("agent_lesson_snapshot_unavailable")
            keywords = tuple(
                str(row["keyword"])
                for row in cursor.execute(
                    "SELECT k.keyword FROM note_keywords AS nk "
                    "JOIN keywords AS k ON k.id = nk.keyword_id "
                    "WHERE nk.note_id = ? AND k.deleted = 0 "
                    "ORDER BY k.keyword COLLATE BINARY, k.id",
                    (note_id,),
                ).fetchall()
            )
            organization = db._library_organization_for_notes(cursor, [note_id])[
                note_id
            ]
            receipt = cursor.execute(
                "SELECT state, note_version, organization_version, "
                "requested_keywords_json "
                "FROM note_organization_receipts WHERE note_id = ? LIMIT 1",
                (note_id,),
            ).fetchone()
            return {
                "note_id": note_id,
                "note_version": int(note["version"]),
                "keywords": keywords,
                "organization_version": str(organization["organization_version"]),
                "receipt_state": str(receipt["state"]) if receipt else None,
                "receipt_note_version": (
                    int(receipt["note_version"]) if receipt else None
                ),
                "receipt_organization_version": (
                    str(receipt["organization_version"]) if receipt else None
                ),
                "receipt_requested_keywords": (
                    tuple(
                        item
                        for item in json.loads(
                            str(receipt["requested_keywords_json"])
                        )
                        if isinstance(item, str)
                    )
                    if receipt
                    else ()
                ),
            }

    def list_notes(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        start_time = time.time()
        log_counter(
            "notes_library_list_notes_attempt",
            labels={"limit": str(limit), "has_offset": str(offset > 0)},
        )

        db = self._get_db(user_id)
        # If notes are truly global and not per-user within the DB, then user_id here doesn't filter.
        # If notes *are* associated with a client_id in the DB table, then CharactersRAGDB.list_notes
        # would need to be modified to filter by its self.client_id (which is user_id here).
        # Assuming current list_notes in ChaChaNotes_DB lists all non-deleted notes.
        results = db.list_notes(limit=limit, offset=offset)

        # Log metrics
        duration = time.time() - start_time
        log_histogram("notes_library_list_notes_duration", duration)
        log_histogram("notes_library_list_notes_count", len(results))
        log_counter(
            "notes_library_list_notes_success", labels={"count": str(len(results))}
        )

        return results

    def count_notes(self, user_id: str) -> int:
        """Count all non-deleted notes in the user's database.

        Args:
            user_id: The user whose per-user database to count in (resolves
                the DB handle only; notes are not per-user-filtered).

        Returns:
            The exact number of non-deleted notes, per
            ``CharactersRAGDB.count_notes``.
        """
        start_time = time.time()
        log_counter("notes_library_count_notes_attempt")

        db = self._get_db(user_id)
        # Same "notes are global, not per-user-filtered" assumption as list_notes above.
        result = db.count_notes()

        duration = time.time() - start_time
        log_histogram("notes_library_count_notes_duration", duration)
        log_counter("notes_library_count_notes_success")

        return result

    # --- Library read seams (task-1337) ---

    def list_library_notes(
        self, user_id: str, *, limit: int = 20, offset: int = 0
    ) -> Dict[str, Any]:
        """Page the active local notes library for agent-facing list tools.

        Args:
            user_id: User scope used to resolve the local notes database.
            limit: Maximum number of notes to return.
            offset: Number of notes to skip.

        Returns:
            A bounded page containing items, exact total, offset, and limit.

        Raises:
            CharactersRAGDBError: If the local notes database cannot be read.
        """
        start_time = time.time()
        log_counter("notes_library_list_library_notes_attempt")

        db = self._get_db(user_id)
        payload = db.list_library_notes_page(limit=limit, offset=offset)

        duration = time.time() - start_time
        log_histogram("notes_library_list_library_notes_duration", duration)
        log_counter(
            "notes_library_list_library_notes_success",
            labels={"count": str(len(payload["items"]))},
        )
        return {
            "items": payload["items"],
            "total": payload["total"],
            "offset": offset,
            "limit": limit,
        }

    def search_library_notes(
        self,
        user_id: str,
        *,
        query: Optional[str] = None,
        folder_sync_id: Optional[str] = None,
        folder: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search the active local notes library for agent-facing tools.

        Args:
            user_id: User scope used to resolve the local notes database.
            query: Optional literal case-insensitive search text.
            folder_sync_id: Optional stable public folder UUID.
            folder: Optional exact portable relative folder path.
            keyword: Optional spelling-exact whole keyword.
            limit: Maximum number of notes to return.
            offset: Number of matching notes to skip.

        Returns:
            A bounded page with exact total and match evidence.

        Raises:
            NotesOrganizationRepositoryError: If a folder selector is invalid,
                missing, ambiguous, deleted, or conflicts with the other form.
            ValueError: If no selector is supplied.
            CharactersRAGDBError: If the local notes database cannot be read.
        """
        start_time = time.time()
        log_counter("notes_library_search_library_notes_attempt")

        db = self._get_db(user_id)
        if folder is None and folder_sync_id is None:
            payload = db.search_library_notes_page(
                **self._library_search_args(
                    query=query,
                    folder_sync_id=None,
                    keyword=keyword,
                    limit=limit,
                    offset=offset,
                )
            )
        else:
            with db.transaction() as connection:
                resolved_folder_sync_id: Optional[str] = None
                if folder is not None:
                    resolved_folder_sync_id = self._resolve_portable_folder_sync_id(
                        connection, folder
                    )
                if folder_sync_id is not None:
                    requested_folder_sync_id = self._active_portable_folder_sync_id(
                        connection, folder_sync_id
                    )
                    if (
                        resolved_folder_sync_id is not None
                        and resolved_folder_sync_id != requested_folder_sync_id
                    ):
                        raise NotesOrganizationRepositoryError(
                            "folder_filter_conflict",
                            "folder and folder_sync_id identify different folders",
                        )
                    resolved_folder_sync_id = requested_folder_sync_id
                payload = db.search_library_notes_page(
                    **self._library_search_args(
                        query=query,
                        folder_sync_id=resolved_folder_sync_id,
                        keyword=keyword,
                        limit=limit,
                        offset=offset,
                    )
                )

        duration = time.time() - start_time
        log_histogram("notes_library_search_library_notes_duration", duration)
        log_counter(
            "notes_library_search_library_notes_success",
            labels={"count": str(len(payload["items"]))},
        )
        return {
            "items": payload["items"],
            "total": payload["total"],
            "offset": offset,
            "limit": limit,
        }

    @staticmethod
    def _library_search_args(
        *,
        query: Optional[str],
        folder_sync_id: Optional[str],
        keyword: Optional[str],
        limit: int,
        offset: int,
    ) -> Dict[str, Any]:
        """Build the compatible DB call without forwarding absent selectors."""

        search_args: Dict[str, Any] = {"limit": limit, "offset": offset}
        if query is not None:
            search_args["query"] = query
        if folder_sync_id is not None:
            search_args["folder_sync_id"] = folder_sync_id
        if keyword is not None:
            search_args["keyword"] = keyword
        return search_args

    @staticmethod
    def _active_portable_folder_sync_id(
        connection: sqlite3.Connection, folder_sync_id: str
    ) -> str:
        """Validate a public folder identity and its complete active lineage."""

        from tldw_chatbook.Sync_Interop.notes_organization import (
            validate_resource_sync_id,
        )

        try:
            validated = validate_resource_sync_id(folder_sync_id.strip())
        except (AttributeError, ValueError) as exc:
            raise NotesOrganizationRepositoryError(
                "invalid_folder_id", "folder_sync_id must be a canonical UUIDv4"
            ) from exc
        row = connection.execute(
            "SELECT id, parent_id, deleted FROM note_folders "
            "WHERE sync_id = ? AND deleted = 0",
            (validated,),
        ).fetchone()
        if row is None:
            raise NotesOrganizationRepositoryError(
                "folder_not_found", "portable folder is missing or deleted"
            )
        visited: set[str] = set()
        current: Optional[sqlite3.Row] = row
        while current is not None:
            current_id = str(current["id"])
            if current_id in visited or bool(current["deleted"]):
                raise NotesOrganizationRepositoryError(
                    "folder_not_found", "portable folder lineage is not active"
                )
            visited.add(current_id)
            parent_id = current["parent_id"]
            if parent_id is None:
                break
            current = connection.execute(
                "SELECT id, parent_id, deleted FROM note_folders WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if current is None:
                raise NotesOrganizationRepositoryError(
                    "folder_not_found", "portable folder lineage is incomplete"
                )
        return validated

    @staticmethod
    def _resolve_portable_folder_sync_id(
        connection: sqlite3.Connection, relative_path: str
    ) -> str:
        """Resolve an exact relative path using server casefold-only rules."""

        if not isinstance(relative_path, str) or relative_path.startswith("/"):
            raise NotesOrganizationRepositoryError(
                "invalid_path", "folder must be a relative portable path"
            )
        try:
            target = portable_relative_path(tuple(relative_path.split("/")))
        except NotesOrganizationRepositoryError as exc:
            raise NotesOrganizationRepositoryError(
                "invalid_path", "folder must be a valid relative portable path"
            ) from exc

        rows = connection.execute(
            "SELECT id, parent_id, name, sync_id, deleted FROM note_folders"
        ).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        matches: List[str] = []
        for row in rows:
            if bool(row["deleted"]) or row["sync_id"] is None:
                continue
            lineage: List[str] = []
            current: Optional[sqlite3.Row] = row
            visited: set[str] = set()
            valid = True
            while current is not None:
                current_id = str(current["id"])
                if current_id in visited or bool(current["deleted"]):
                    valid = False
                    break
                visited.add(current_id)
                lineage.append(str(current["name"]))
                parent_id = current["parent_id"]
                if parent_id is None:
                    current = None
                else:
                    current = by_id.get(str(parent_id))
                    if current is None:
                        valid = False
                        break
            if not valid:
                continue
            try:
                candidate = portable_relative_path(tuple(reversed(lineage)))
            except NotesOrganizationRepositoryError:
                continue
            if candidate == target:
                matches.append(str(row["sync_id"]))

        unique_matches = sorted(set(matches))
        if not unique_matches:
            raise NotesOrganizationRepositoryError(
                "folder_not_found", "portable folder path is missing or deleted"
            )
        if len(unique_matches) != 1:
            raise NotesOrganizationRepositoryError(
                "ambiguous_path", "portable folder path is ambiguous"
            )
        return unique_matches[0]

    def get_library_note_text(
        self, user_id: str, note_id: str, *, start: int = 0, max_chars: int = 8000
    ) -> Optional[Dict[str, Any]]:
        """Read a windowed text segment for one active note.

        Args:
            user_id: User scope used to resolve the local notes database.
            note_id: Stable note identifier.
            start: Zero-based character offset.
            max_chars: Maximum characters to return.

        Returns:
            Bounded note metadata and text, or None when absent.

        Raises:
            CharactersRAGDBError: If the local notes database cannot be read.
        """
        start_time = time.time()
        log_counter("notes_library_get_library_note_text_attempt")

        db = self._get_db(user_id)
        detail = db.get_library_note_text(note_id, start=start, max_chars=max_chars)

        duration = time.time() - start_time
        log_histogram("notes_library_get_library_note_text_duration", duration)
        log_counter(
            "notes_library_get_library_note_text_result",
            labels={"found": str(detail is not None)},
        )
        return detail

    def save_note_with_organization(
        self,
        user_id: str,
        *,
        title: str,
        content: str,
        note_id: Optional[str] = None,
        expected_version: Optional[int] = None,
        ensure_keywords: Sequence[str] = (),
        folder_sync_id: Optional[str] = None,
        folder: Optional[str] = None,
        expected_organization_version: Optional[str] = None,
        receipt_id: Optional[str] = None,
        server_profile_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        _agent_lesson_context: object | None = None,
        _agent_lesson_raw_arguments: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Save note content and additive organization in one Notes transaction."""

        if (note_id is None) != (expected_version is None):
            raise ValueError("note_id and expected_version must be supplied together")
        if folder is not None and folder_sync_id is not None:
            raise ValueError("folder and folder_sync_id are alternative inputs")
        if folder is not None:
            portable_collision_key(folder, maximum=255)
            folder = folder.strip()
        if folder_sync_id is not None:
            from tldw_chatbook.Sync_Interop.notes_organization import (
                validate_resource_sync_id,
            )

            folder_sync_id = validate_resource_sync_id(folder_sync_id.strip())
        requested_keywords = self._normalize_ensured_keywords(ensure_keywords)
        organization_requested = bool(requested_keywords or folder or folder_sync_id)
        if note_id is not None and organization_requested and not expected_organization_version:
            raise ValueError(
                "expected_organization_version is required for organization-changing updates"
            )
        stable_receipt_id = str(receipt_id or uuid.uuid4()).strip()
        if not stable_receipt_id or len(stable_receipt_id) > 512:
            raise ValueError("receipt_id must be non-blank text")

        db = self._get_db(user_id)
        folders = LocalNoteFolderRepository(db)
        with self.note_transaction(user_id) as cursor:
            authority_note = None
            authority_keywords: tuple[str, ...] = ()
            authority_organization_version = None
            authority_receipt = None
            if note_id is not None:
                authority_note = cursor.execute(
                    "SELECT version FROM notes WHERE id = ? AND deleted = 0",
                    (note_id,),
                ).fetchone()
                authority_keywords = tuple(
                    str(row["keyword"])
                    for row in cursor.execute(
                        "SELECT k.keyword FROM note_keywords AS nk "
                        "JOIN keywords AS k ON k.id = nk.keyword_id "
                        "WHERE nk.note_id = ? AND k.deleted = 0 "
                        "ORDER BY k.keyword COLLATE BINARY, k.id",
                        (note_id,),
                    ).fetchall()
                )
                if authority_note is not None:
                    authority_organization_version = str(
                        db._library_organization_for_notes(cursor, [note_id])[
                            note_id
                        ]["organization_version"]
                    )
                authority_receipt = cursor.execute(
                    "SELECT state, note_version, organization_version, "
                    "requested_keywords_json FROM note_organization_receipts "
                    "WHERE note_id = ? LIMIT 1",
                    (note_id,),
                ).fetchone()

            from tldw_chatbook.Notes.agent_lessons import (
                classify_agent_lesson,
                classify_lesson_credentials,
            )

            receipt_state = (
                str(authority_receipt["state"]) if authority_receipt else None
            )
            classification = classify_agent_lesson(
                requested_keywords=requested_keywords,
                current_keywords=authority_keywords,
                receipt_state=receipt_state,
            )
            approved_attempt = False
            if _agent_lesson_context is not None:
                try:
                    from tldw_chatbook.Agents.library_tool_provider import (
                        _AgentLessonApprovalAuthority,
                        _AgentLessonAuthorityRefusal,
                        _AgentLessonMutationContext,
                        LibraryToolProvider,
                    )

                    approved_attempt = (
                        type(_agent_lesson_context) is _AgentLessonMutationContext
                        and type(_agent_lesson_context.authority)
                        is _AgentLessonApprovalAuthority
                        and type(_agent_lesson_context.issuer)
                        is LibraryToolProvider
                    )
                except ImportError:
                    approved_attempt = False
            if _agent_lesson_context is not None and (
                classification.is_agent_lesson or approved_attempt
            ):
                if not approved_attempt:
                    reason = (
                        "foreground_required"
                        if type(_agent_lesson_context) is _AgentLessonMutationContext
                        and _agent_lesson_context.actor is not None
                        and _agent_lesson_context.actor.kind != "primary"
                        else "approval_required"
                    )
                    raise NotesOrganizationRepositoryError(reason, reason)
                credential_check = classify_lesson_credentials(content)
                if not credential_check.accepted:
                    raise NotesOrganizationRepositoryError(
                        "credential_material_detected",
                        "credential_material_detected",
                    )
                receipt_keywords = (
                    tuple(
                        item
                        for item in json.loads(
                            str(authority_receipt["requested_keywords_json"])
                        )
                        if isinstance(item, str)
                    )
                    if authority_receipt
                    else ()
                )
                receipt_version = (
                    f"{int(authority_receipt['note_version'])}:"
                    f"{str(authority_receipt['organization_version'])}"
                    if authority_receipt
                    else None
                )
                try:
                    _agent_lesson_context.issuer._consume_agent_lesson_approval(
                        _agent_lesson_context,
                        raw_arguments=_agent_lesson_raw_arguments or {},
                        note_id=note_id,
                        classification=classification,
                        requested_marker="agent-lesson" in requested_keywords,
                        current_marker="agent-lesson" in authority_keywords,
                        receipt_requested_marker="agent-lesson" in receipt_keywords,
                        observed_note_version=(
                            int(authority_note["version"])
                            if authority_note is not None
                            else None
                        ),
                        observed_organization_version=authority_organization_version,
                        receipt_state=receipt_state,
                        receipt_version=receipt_version,
                    )
                except _AgentLessonAuthorityRefusal as exc:
                    raise NotesOrganizationRepositoryError(
                        exc.reason_code, exc.reason_code
                    ) from None

            profile, dataset, ready = self._organization_scope(
                cursor,
                server_profile_id=server_profile_id,
                dataset_id=dataset_id,
            )
            retry = cursor.execute(
                "SELECT * FROM note_organization_receipts WHERE receipt_id = ?",
                (stable_receipt_id,),
            ).fetchone()
            unresolved_receipt = None
            if note_id is not None:
                unresolved_receipt = cursor.execute(
                    "SELECT * FROM note_organization_receipts WHERE note_id = ?",
                    (note_id,),
                ).fetchone()
            if unresolved_receipt is not None:
                stored_keywords = self._receipt_requested_keywords(
                    unresolved_receipt
                )
                requested_keywords = tuple(
                    dict.fromkeys((*stored_keywords, *requested_keywords))
                )
                if folder is None and folder_sync_id is None:
                    stored_folder_name = unresolved_receipt[
                        "requested_folder_name"
                    ]
                    stored_folder_sync_id = unresolved_receipt[
                        "requested_folder_sync_id"
                    ]
                    folder = (
                        str(stored_folder_name)
                        if stored_folder_name is not None
                        else None
                    )
                    folder_sync_id = (
                        str(stored_folder_sync_id)
                        if stored_folder_sync_id is not None
                        else None
                    )
            organization_requested = bool(
                requested_keywords or folder or folder_sync_id
            )
            if (
                note_id is not None
                and organization_requested
                and not expected_organization_version
            ):
                raise ValueError(
                    "expected_organization_version is required for "
                    "organization-changing updates"
                )
            request_binding = self._organization_receipt_request_binding(
                title=title,
                content=content,
                note_id=note_id,
                expected_version=expected_version,
                folder=folder,
                folder_sync_id=folder_sync_id,
                keywords=requested_keywords,
                expected_organization_version=expected_organization_version,
                profile=profile,
                dataset=dataset,
            )
            if retry is not None:
                self._validate_receipt_retry(
                    retry,
                    request_binding=request_binding,
                )
                self._validate_receipt_target(cursor, retry)
                return self._organization_save_result(
                    db, cursor, str(retry["note_id"]), str(retry["state"])
                )
            completed = cursor.execute(
                "SELECT * FROM note_sync_publication_intents WHERE intent_id = ?",
                (stable_receipt_id,),
            ).fetchone()
            if completed is not None:
                self._validate_completed_receipt_retry(
                    cursor,
                    completed,
                    request_binding=request_binding,
                )
                return self._organization_save_result(
                    db, cursor, str(completed["note_id"]), None
                )
            if unresolved_receipt is not None and self._receipt_matches_request(
                unresolved_receipt, request_binding=request_binding
            ):
                self._validate_receipt_target(cursor, unresolved_receipt)
                return self._organization_save_result(
                    db,
                    cursor,
                    str(unresolved_receipt["note_id"]),
                    str(unresolved_receipt["state"]),
                )
            current_note = None
            if note_id is not None:
                current_note = cursor.execute(
                    "SELECT * FROM notes WHERE id = ? AND deleted = 0", (note_id,)
                ).fetchone()
                if current_note is None or int(current_note["version"]) != expected_version:
                    raise ConflictError(
                        "Note content changed before the organization save.",
                        entity="notes",
                        entity_id=note_id,
                    )
                current_organization = db._library_organization_for_notes(
                    cursor, [note_id]
                )[note_id]["organization_version"]
                if (
                    organization_requested
                    and current_organization != expected_organization_version
                ):
                    raise NotesOrganizationRepositoryError(
                        "organization_changed",
                        "note organization changed before the save",
                    )
                if unresolved_receipt is not None:
                    if int(unresolved_receipt["note_version"]) != expected_version:
                        raise ConflictError(
                            "Receipt content changed before the organization save.",
                            entity="notes",
                            entity_id=note_id,
                        )
                    if (
                        str(unresolved_receipt["organization_version"])
                        != expected_organization_version
                    ):
                        raise NotesOrganizationRepositoryError(
                            "organization_changed",
                            "receipt organization changed before the save",
                        )
                    stable_receipt_id = str(unresolved_receipt["receipt_id"])

            keyword_conflict = self._keyword_identity_conflict(
                cursor,
                requested_keywords,
                profile=profile,
                dataset=dataset,
            )
            pending = organization_requested and (
                not ready
                or keyword_conflict
                or (
                    unresolved_receipt is not None
                    and unresolved_receipt["state"] == "pending_organization"
                )
            )
            selected_note_id = note_id or str(uuid.uuid4())
            if current_note is None:
                db._add_note_with_cursor(
                    cursor,
                    title=title,
                    content=content,
                    note_id=selected_note_id,
                )
                note_version = 1
            else:
                db._update_note_with_cursor(
                    cursor,
                    note_id=selected_note_id,
                    update_data={"title": title, "content": content},
                    expected_version=int(expected_version),
                )
                note_version = int(expected_version) + 1
            self._inject_organization_failure("after_note_write")

            if pending:
                # The normal note trigger is the publication intent for ready
                # saves. A blocking receipt owns this pending content instead.
                cursor.execute(
                    "DELETE FROM sync_log WHERE entity = 'notes' AND entity_id = ? "
                    "AND version = ?",
                    (selected_note_id, note_version),
                )
                self._write_organization_receipt(
                    db,
                    cursor,
                    receipt_id=stable_receipt_id,
                    note_id=selected_note_id,
                    note_version=note_version,
                    state="pending_organization",
                    requested_folder_name=folder,
                    requested_folder_sync_id=folder_sync_id,
                    requested_keywords=requested_keywords,
                    request_binding=request_binding,
                )
                self._inject_organization_failure("after_receipt")
                return self._organization_save_result(
                    db, cursor, selected_note_id, "pending_organization"
                )

            folder_row, folder_created, folder_collisions = self._ensure_save_folder(
                cursor,
                folder=folder,
                folder_sync_id=folder_sync_id,
            )
            self._inject_organization_failure("after_folder_ensure")

            keyword_rows: List[sqlite3.Row] = []
            created_keyword_rows: List[sqlite3.Row] = []
            for keyword in requested_keywords:
                row = cursor.execute(
                    "SELECT * FROM keywords WHERE keyword = ? COLLATE BINARY "
                    "AND deleted = 0",
                    (keyword,),
                ).fetchone()
                if row is None:
                    keyword_id = db.add_keyword(keyword, cursor=cursor)
                    row = cursor.execute(
                        "SELECT * FROM keywords WHERE id = ?", (keyword_id,)
                    ).fetchone()
                    created_keyword_rows.append(row)
                keyword_rows.append(row)
            self._inject_organization_failure("after_keyword_ensure")

            folder_link_created = False
            if folder_row is not None and not folder_collisions:
                before = cursor.execute(
                    "SELECT 1 FROM note_folder_memberships WHERE folder_id = ? "
                    "AND note_id = ? AND deleted = 0 AND ownership = 'manual' LIMIT 1",
                    (folder_row["id"], selected_note_id),
                ).fetchone()
                folders.attach_manual(
                    folder_id=str(folder_row["id"]),
                    note_id=selected_note_id,
                    cursor=cursor,
                )
                folder_link_created = before is None
            keyword_links: List[sqlite3.Row] = []
            for row in keyword_rows:
                if db.link_note_to_keyword(
                    selected_note_id, int(row["id"]), cursor=cursor
                ):
                    keyword_links.append(row)
            self._inject_organization_failure("after_membership")

            if profile is not None and dataset is not None:
                from tldw_chatbook.Notes.notes_organization_repository import (
                    NotesOrganizationRepository,
                )

                organization = NotesOrganizationRepository(
                    db, server_profile_id=profile
                )
                if folder_created and folder_row is not None:
                    self._record_organization_intent(
                        organization,
                        cursor,
                        profile=profile,
                        dataset=dataset,
                        domain="notes.folder",
                        object_id=str(folder_row["sync_id"]),
                        payload={"name": str(folder_row["name"]), "parent_sync_id": None},
                        source_version=int(folder_row["version"]),
                    )
                for row in created_keyword_rows:
                    self._record_organization_intent(
                        organization,
                        cursor,
                        profile=profile,
                        dataset=dataset,
                        domain="notes.keyword",
                        object_id=str(row["sync_id"]),
                        payload={"keyword": str(row["keyword"])},
                        source_version=int(row["version"]),
                    )
                if folder_link_created and folder_row is not None:
                    payload = {
                        "note_id": selected_note_id,
                        "folder_sync_id": str(folder_row["sync_id"]),
                    }
                    self._record_organization_link_intent(
                        organization,
                        cursor,
                        profile=profile,
                        dataset=dataset,
                        domain="notes.folder_link",
                        members=(selected_note_id, str(folder_row["sync_id"])),
                        payload=payload,
                    )
                for row in keyword_links:
                    payload = {
                        "subject_type": "note",
                        "subject_id": selected_note_id,
                        "keyword_sync_id": str(row["sync_id"]),
                    }
                    self._record_organization_link_intent(
                        organization,
                        cursor,
                        profile=profile,
                        dataset=dataset,
                        domain="notes.keyword_link",
                        members=("note", selected_note_id, str(row["sync_id"])),
                        payload=payload,
                    )
            self._inject_organization_failure("after_intent")

            receipt_state: Optional[str] = None
            if folder_collisions:
                review_id = self._ensure_folder_placement_review(
                    cursor,
                    profile=profile or "local-only",
                    dataset=dataset or "local-only",
                    requested_name=str(folder),
                    collision_ids=folder_collisions,
                )
                self._write_organization_receipt(
                    db,
                    cursor,
                    receipt_id=stable_receipt_id,
                    note_id=selected_note_id,
                    note_version=note_version,
                    state="placement_review",
                    requested_folder_name=folder,
                    requested_folder_sync_id=folder_sync_id,
                    requested_keywords=requested_keywords,
                    request_binding=request_binding,
                    review_id=review_id,
                    collision_ids=folder_collisions,
                )
                if (
                    unresolved_receipt is not None
                    and unresolved_receipt["state"] == "placement_review"
                    and unresolved_receipt["review_id"] != review_id
                ):
                    self._resolve_obsolete_placement_review(
                        cursor, str(unresolved_receipt["review_id"])
                    )
                receipt_state = "placement_review"
                self._inject_organization_failure("after_receipt")
            elif (
                unresolved_receipt is not None
                and unresolved_receipt["state"] == "placement_review"
            ):
                cursor.execute(
                    "DELETE FROM note_organization_receipts "
                    "WHERE receipt_id = ? AND note_id = ?",
                    (str(unresolved_receipt["receipt_id"]), selected_note_id),
                )
                self._resolve_obsolete_placement_review(
                    cursor, str(unresolved_receipt["review_id"])
                )
                self._inject_organization_failure("after_receipt")
            return self._organization_save_result(
                db, cursor, selected_note_id, receipt_state
            )

    @staticmethod
    def _normalize_ensured_keywords(values: Sequence[str]) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError("ensure_keywords must be a sequence of keywords")
        normalized: List[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("ensure_keywords contains an invalid keyword")
            text = value.strip()
            if text not in normalized:
                normalized.append(text)
        return tuple(normalized)

    def _inject_organization_failure(self, stage: str) -> None:
        if self._organization_failure_injector is not None:
            self._organization_failure_injector(stage)

    @staticmethod
    def _organization_scope(
        cursor: sqlite3.Cursor,
        *,
        server_profile_id: Optional[str],
        dataset_id: Optional[str],
    ) -> tuple[Optional[str], Optional[str], bool]:
        if (server_profile_id is None) != (dataset_id is None):
            raise ValueError("server_profile_id and dataset_id must be supplied together")
        if server_profile_id is None:
            rows = cursor.execute(
                "SELECT * FROM notes_organization_sync_checkpoints"
            ).fetchall()
            if not rows:
                return None, None, True
            if len(rows) != 1:
                raise NotesOrganizationRepositoryError(
                    "ambiguous_profile", "multiple Notes organization profiles require routing"
                )
            row = rows[0]
            profile = str(row["server_profile_id"])
            dataset = str(row["dataset_id"])
        else:
            profile = server_profile_id.strip()
            dataset = dataset_id.strip()  # type: ignore[union-attr]
            if not profile or not dataset:
                raise ValueError("profile and dataset must be non-blank")
            row = cursor.execute(
                "SELECT * FROM notes_organization_sync_checkpoints "
                "WHERE server_profile_id = ? AND dataset_id = ?",
                (profile, dataset),
            ).fetchone()
            if row is None:
                return profile, dataset, False
        open_review = cursor.execute(
            "SELECT 1 FROM notes_organization_adoption_reviews AS review WHERE "
            "server_profile_id = ? AND dataset_id = ? AND state = 'open' "
            "AND NOT EXISTS (SELECT 1 FROM note_organization_receipts AS receipt "
            "WHERE receipt.review_id = review.review_id "
            "AND receipt.state = 'placement_review') LIMIT 1",
            (profile, dataset),
        ).fetchone()
        ready = (
            row["local_state"] == "ready"
            and row["server_state"] == "ready"
            and row["inventory_phase"] == "complete"
            and row["error_code"] is None
            and open_review is None
        )
        return profile, dataset, ready

    @staticmethod
    def _keyword_identity_conflict(
        cursor: sqlite3.Cursor,
        keywords: Sequence[str],
        *,
        profile: Optional[str],
        dataset: Optional[str],
    ) -> bool:
        rows = cursor.execute(
            "SELECT keyword FROM keywords WHERE deleted = 0"
        ).fetchall()
        existing = [str(row["keyword"]) for row in rows]
        if any(
            any(current.casefold() == requested.casefold() for current in existing)
            and requested not in existing
            for requested in keywords
        ):
            return True
        if profile is None or dataset is None:
            return False
        review_keys = {
            str(row["collision_key"])
            for row in cursor.execute(
                "SELECT collision_key FROM notes_organization_adoption_reviews "
                "WHERE server_profile_id = ? AND dataset_id = ? "
                "AND domain = 'notes.keyword' AND state = 'open'",
                (profile, dataset),
            ).fetchall()
        }
        return any(requested.casefold() in review_keys for requested in keywords)

    @staticmethod
    def _ensure_save_folder(
        cursor: sqlite3.Cursor,
        *,
        folder: Optional[str],
        folder_sync_id: Optional[str],
    ) -> tuple[Optional[sqlite3.Row], bool, tuple[str, ...]]:
        if folder_sync_id is not None:
            sync_id = NotesInteropService._active_portable_folder_sync_id(
                cursor, folder_sync_id
            )
            row = cursor.execute(
                "SELECT * FROM note_folders WHERE sync_id = ? AND deleted = 0",
                (sync_id,),
            ).fetchone()
            return row, False, ()
        if folder is None:
            return None, False, ()
        display = folder.strip()
        key = portable_collision_key(display, maximum=255)
        matches = [
            row
            for row in cursor.execute(
                "SELECT * FROM note_folders WHERE parent_id IS NULL AND deleted = 0"
            ).fetchall()
            if str(row["name"]).casefold() == key
        ]
        if matches:
            exact = [row for row in matches if str(row["name"]) == display]
            if exact and exact[0]["sync_id"]:
                return exact[0], False, ()
            collisions = tuple(sorted(str(row["id"]) for row in matches))
            return exact[0] if exact else matches[0], False, collisions

        local_key = unicodedata.normalize("NFKC", display).casefold()
        normalized_path = f"/{local_key}"
        local_collisions = cursor.execute(
            "SELECT * FROM note_folders WHERE normalized_path = ? AND deleted = 0 "
            "ORDER BY id",
            (normalized_path,),
        ).fetchall()
        if local_collisions:
            return (
                local_collisions[0],
                False,
                tuple(str(row["id"]) for row in local_collisions),
            )

        local_id = str(uuid.uuid4())
        sync_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat(timespec="milliseconds")
        cursor.execute(
            "INSERT INTO note_folders("
            "id, parent_id, name, normalized_name, path, normalized_path, "
            "version, deleted, created_at, modified_at, sync_id"
            ") VALUES (?, NULL, ?, ?, ?, ?, 1, 0, ?, ?, ?)",
            (
                local_id,
                display,
                local_key,
                f"/{display}",
                normalized_path,
                now,
                now,
                sync_id,
            ),
        )
        row = cursor.execute(
            "SELECT * FROM note_folders WHERE id = ?", (local_id,)
        ).fetchone()
        return row, True, ()

    @staticmethod
    def _record_organization_intent(
        repository: NotesOrganizationRepository,
        cursor: sqlite3.Cursor,
        *,
        profile: str,
        dataset: str,
        domain: str,
        object_id: str,
        payload: Mapping[str, object],
        source_version: int,
    ) -> None:
        repository._record_inferred_intent_with_cursor(
            cursor,
            profile=profile,
            dataset=dataset,
            domain=domain,
            object_id=object_id,
            operation="upsert",
            payload=payload,
            source_version=source_version,
        )

    @staticmethod
    def _record_organization_link_intent(
        repository: NotesOrganizationRepository,
        cursor: sqlite3.Cursor,
        *,
        profile: str,
        dataset: str,
        domain: str,
        members: Sequence[str],
        payload: Mapping[str, object],
    ) -> None:
        from tldw_chatbook.Sync_Interop.notes_organization import organization_link_id

        object_id = organization_link_id(domain, members)
        repository._record_inferred_intent_with_cursor(
            cursor,
            profile=profile,
            dataset=dataset,
            domain=domain,
            object_id=object_id,
            operation="upsert",
            payload=payload,
        )

    @staticmethod
    def _write_organization_receipt(
        db: CharactersRAGDB,
        cursor: sqlite3.Cursor,
        *,
        receipt_id: str,
        note_id: str,
        note_version: int,
        state: str,
        requested_folder_name: Optional[str],
        requested_folder_sync_id: Optional[str],
        requested_keywords: Sequence[str],
        request_binding: Mapping[str, object],
        review_id: Optional[str] = None,
        collision_ids: Sequence[str] = (),
    ) -> None:
        now = db._get_current_utc_timestamp_iso()
        initial_version = db._library_organization_for_notes(cursor, [note_id])[
            note_id
        ]["organization_version"]
        values = (
            requested_folder_name.strip() if requested_folder_name else None,
            requested_folder_sync_id,
            NotesInteropService._serialize_receipt_request(
                requested_keywords, request_binding
            ),
            review_id,
            json.dumps(list(collision_ids), separators=(",", ":")),
            note_version,
            initial_version,
            state,
            now,
        )
        existing = cursor.execute(
            "SELECT 1 FROM note_organization_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
        if existing is None:
            cursor.execute(
                """
                INSERT INTO note_organization_receipts(
                    receipt_id, note_id, requested_folder_name,
                    requested_folder_sync_id, requested_keywords_json, review_id,
                    collision_ids_json, note_version, organization_version, state,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (receipt_id, note_id, *values, now),
            )
        else:
            cursor.execute(
                """
                UPDATE note_organization_receipts
                   SET requested_folder_name = ?, requested_folder_sync_id = ?,
                       requested_keywords_json = ?, review_id = ?,
                       collision_ids_json = ?, note_version = ?,
                       organization_version = ?, state = ?, updated_at = ?
                 WHERE receipt_id = ? AND note_id = ?
                """,
                (*values, receipt_id, note_id),
            )
        receipt_version = db._library_organization_for_notes(cursor, [note_id])[
            note_id
        ]["organization_version"]
        cursor.execute(
            "UPDATE note_organization_receipts SET organization_version = ? "
            "WHERE receipt_id = ?",
            (receipt_version, receipt_id),
        )

    @staticmethod
    def _ensure_folder_placement_review(
        cursor: sqlite3.Cursor,
        *,
        profile: str,
        dataset: str,
        requested_name: str,
        collision_ids: Sequence[str],
    ) -> str:
        local_id = str(collision_ids[0])
        existing = cursor.execute(
            "SELECT review_id FROM notes_organization_adoption_reviews WHERE "
            "server_profile_id = ? AND dataset_id = ? AND domain = 'notes.folder' "
            "AND local_object_id = ?",
            (profile, dataset, local_id),
        ).fetchone()
        if existing is not None:
            review_id = str(existing["review_id"])
            now = datetime.now(UTC).isoformat(timespec="milliseconds")
            cursor.execute(
                "UPDATE notes_organization_adoption_reviews SET "
                "collision_key = ?, display_name = ?, portable_path = ?, "
                "state = 'open', resolution = NULL, resolved_at = NULL, "
                "updated_at = ? WHERE review_id = ?",
                (
                    portable_collision_key(requested_name),
                    requested_name.strip(),
                    portable_collision_key(requested_name),
                    now,
                    review_id,
                ),
            )
            return review_id
        review_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"tldw:note-placement:{profile}:{dataset}:{local_id}",
            )
        )
        now = datetime.now(UTC).isoformat(timespec="milliseconds")
        cursor.execute(
            """
            INSERT INTO notes_organization_adoption_reviews(
                review_id, server_profile_id, dataset_id, domain,
                local_object_id, remote_object_id, collision_key, display_name,
                portable_path, state, created_at, updated_at
            ) VALUES (?, ?, ?, 'notes.folder', ?, NULL, ?, ?, ?, 'open', ?, ?)
            """,
            (
                review_id,
                profile,
                dataset,
                local_id,
                portable_collision_key(requested_name),
                requested_name.strip(),
                portable_collision_key(requested_name),
                now,
                now,
            ),
        )
        return review_id

    @staticmethod
    def _resolve_obsolete_placement_review(
        cursor: sqlite3.Cursor, review_id: str
    ) -> None:
        now = datetime.now(UTC).isoformat(timespec="milliseconds")
        cursor.execute(
            "UPDATE notes_organization_adoption_reviews SET state = 'resolved', "
            "resolution = 'keep_local', resolved_at = ?, updated_at = ? "
            "WHERE review_id = ? AND state = 'open' AND NOT EXISTS ("
            "SELECT 1 FROM note_organization_receipts "
            "WHERE review_id = ? AND state = 'placement_review')",
            (now, now, review_id, review_id),
        )

    @staticmethod
    def _organization_receipt_request_binding(
        *,
        title: str,
        content: str,
        note_id: Optional[str],
        expected_version: Optional[int],
        folder: Optional[str],
        folder_sync_id: Optional[str],
        keywords: Sequence[str],
        expected_organization_version: Optional[str],
        profile: Optional[str],
        dataset: Optional[str],
    ) -> Dict[str, object]:
        canonical_request = {
            "content": content,
            "dataset_id": dataset,
            "ensure_keywords": list(keywords),
            "expected_organization_version": expected_organization_version,
            "expected_version": expected_version,
            "folder": folder.strip() if folder else None,
            "folder_sync_id": folder_sync_id,
            "note_id": note_id,
            "server_profile_id": profile,
            "title": title,
        }
        serialized = json.dumps(
            canonical_request,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return {
            "dataset_id": dataset,
            "expected_organization_version": expected_organization_version,
            "expected_version": expected_version,
            "fingerprint": hashlib.sha256(serialized).hexdigest(),
            "note_id": note_id,
            "server_profile_id": profile,
        }

    @staticmethod
    def _serialize_receipt_request(
        keywords: Sequence[str], request_binding: Mapping[str, object]
    ) -> str:
        return json.dumps(
            [*keywords, {"_request": dict(request_binding)}],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @staticmethod
    def _receipt_requested_keywords(receipt: sqlite3.Row) -> tuple[str, ...]:
        try:
            stored = json.loads(str(receipt["requested_keywords_json"]))
        except (TypeError, json.JSONDecodeError) as exc:
            raise NotesOrganizationRepositoryError(
                "receipt_conflict", "receipt organization request is invalid"
            ) from exc
        if not isinstance(stored, list):
            raise NotesOrganizationRepositoryError(
                "receipt_conflict", "receipt organization request is invalid"
            )
        return tuple(item for item in stored if isinstance(item, str))

    @staticmethod
    def _validate_receipt_retry(
        receipt: sqlite3.Row,
        *,
        request_binding: Mapping[str, object],
    ) -> None:
        if not NotesInteropService._receipt_matches_request(
            receipt, request_binding=request_binding
        ):
            raise NotesOrganizationRepositoryError(
                "receipt_conflict", "receipt_id is already bound to another request"
            )

    @staticmethod
    def _receipt_matches_request(
        receipt: sqlite3.Row, *, request_binding: Mapping[str, object]
    ) -> bool:
        try:
            stored = json.loads(str(receipt["requested_keywords_json"]))
            stored_binding = stored[-1]["_request"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError):
            return False
        return stored_binding == dict(request_binding)

    @staticmethod
    def _validate_receipt_target(
        cursor: sqlite3.Cursor, receipt: sqlite3.Row
    ) -> None:
        note = cursor.execute(
            "SELECT version FROM notes WHERE id = ? AND deleted = 0",
            (str(receipt["note_id"]),),
        ).fetchone()
        if note is None or int(note["version"]) != int(receipt["note_version"]):
            raise NotesOrganizationRepositoryError(
                "receipt_conflict", "receipt target changed after the original request"
            )

    @staticmethod
    def _validate_completed_receipt_retry(
        cursor: sqlite3.Cursor,
        publication: sqlite3.Row,
        *,
        request_binding: Mapping[str, object],
    ) -> None:
        """Validate a replay against its durable finalized publication identity."""

        if publication["cancelled_at"] is not None:
            raise NotesOrganizationRepositoryError(
                "receipt_conflict", "the original receipt request was cancelled"
            )
        if (
            str(publication["request_fingerprint"])
            != str(request_binding.get("fingerprint") or "")
            or str(publication["server_profile_id"])
            != str(request_binding.get("server_profile_id") or "")
            or str(publication["dataset_id"])
            != str(request_binding.get("dataset_id") or "")
        ):
            raise NotesOrganizationRepositoryError(
                "receipt_conflict", "receipt_id is already bound to another request"
            )
        note = cursor.execute(
            "SELECT version FROM notes WHERE id = ? AND deleted = 0",
            (str(publication["note_id"]),),
        ).fetchone()
        if note is None or int(note["version"]) != int(publication["entity_version"]):
            raise NotesOrganizationRepositoryError(
                "receipt_conflict", "receipt target changed after the original request"
            )

    @staticmethod
    def _organization_save_result(
        db: CharactersRAGDB,
        cursor: sqlite3.Cursor,
        note_id: str,
        receipt_state: Optional[str],
    ) -> Dict[str, Any]:
        note = cursor.execute(
            "SELECT id, title, version FROM notes WHERE id = ? AND deleted = 0",
            (note_id,),
        ).fetchone()
        if note is None:
            raise NotesOrganizationRepositoryError(
                "note_not_found", "receipt note is missing or deleted"
            )
        result: Dict[str, Any] = {
            "id": str(note["id"]),
            "title": str(note["title"]),
            "version": int(note["version"]),
            "receipt_state": receipt_state,
        }
        result.update(db._library_organization_for_notes(cursor, [note_id])[note_id])
        return result

    def update_note(
        self,
        user_id: str,
        note_id: str,
        update_data: Dict[str, Any],
        expected_version: int,
    ) -> bool:
        start_time = time.time()
        log_counter(
            "notes_library_update_note_attempt",
            labels={
                "fields_count": str(len(update_data)),
                "has_title": str("title" in update_data),
                "has_content": str("content" in update_data),
            },
        )

        try:
            db = self._get_db(
                user_id
            )  # The db instance will have user_id as its client_id for the update
            result = db.update_note(
                note_id=note_id,
                update_data=update_data,
                expected_version=expected_version,
            )

            # Log metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_update_note_duration",
                duration,
                labels={"status": "success" if result else "conflict"},
            )
            log_counter(
                "notes_library_update_note_result", labels={"success": str(result)}
            )

            return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_update_note_duration",
                duration,
                labels={"status": "error"},
            )
            log_counter(
                "notes_library_update_note_error",
                labels={"error_type": type(e).__name__},
            )
            raise

    def soft_delete_note(
        self, user_id: str, note_id: str, expected_version: int
    ) -> bool:
        start_time = time.time()
        log_counter("notes_library_delete_note_attempt")

        try:
            db = self._get_db(user_id)  # client_id for operation comes from db instance
            result = db.soft_delete_note(
                note_id=note_id, expected_version=expected_version
            )

            # Log metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_delete_note_duration",
                duration,
                labels={"status": "success" if result else "conflict"},
            )
            log_counter(
                "notes_library_delete_note_result", labels={"success": str(result)}
            )

            return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_delete_note_duration",
                duration,
                labels={"status": "error"},
            )
            log_counter(
                "notes_library_delete_note_error",
                labels={"error_type": type(e).__name__},
            )
            raise

    def restore_note(
        self, user_id: str, note_id: str, expected_version: int
    ) -> bool:
        """Restore a soft-deleted note through the per-user database seam.

        Args:
            user_id: User identity used to select the unified DB client.
            note_id: Stable note identity to restore.
            expected_version: Version of the tombstone being restored.

        Returns:
            ``True`` when the note is restored or is already active.

        Raises:
            ConflictError: If the note is missing or its tombstone is stale.
            CharactersRAGDBError: If the database operation fails.
        """
        start_time = time.time()
        log_counter("notes_library_restore_note_attempt")

        try:
            db = self._get_db(user_id)
            result = db.restore_note(
                note_id=note_id, expected_version=expected_version
            )
            duration = time.time() - start_time
            log_histogram(
                "notes_library_restore_note_duration",
                duration,
                labels={"status": "success" if result else "conflict"},
            )
            log_counter(
                "notes_library_restore_note_result", labels={"success": str(result)}
            )
            return bool(result)
        except Exception as e:
            duration = time.time() - start_time
            log_histogram(
                "notes_library_restore_note_duration",
                duration,
                labels={"status": "error"},
            )
            log_counter(
                "notes_library_restore_note_error",
                labels={"error_type": type(e).__name__},
            )
            raise

    def search_notes(
        self,
        user_id: str,
        search_term: str,
        limit: int = 10,
        fts_match_query: Optional[str] = None,
        *,
        id_allowlist: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search notes, optionally restricted to a caller-provided id allowlist.

        Args:
            user_id: Owning user id (resolves the per-user DB instance).
            search_term: Plain user search text.
            limit: Maximum number of notes to return.
            fts_match_query: Optional pre-built FTS5 MATCH expression.
            id_allowlist: Optional note ids to restrict results to
                (rag-scope narrowing, task-6). ``None`` (the default) is
                unrestricted -- forwarded only when provided so existing
                callers (and test fakes) without the parameter keep working
                unchanged.

        Returns:
            Matching note rows.
        """
        start_time = time.time()
        log_counter(
            "notes_library_search_notes_attempt",
            labels={"search_term_length": str(len(search_term)), "limit": str(limit)},
        )
        try:
            db = self._get_db(user_id)
            # Similar to list_notes, if search should be user-specific, CharactersRAGDB.search_notes needs adjustment.
            # Forward the caller-built MATCH expression / id allowlist only
            # when provided so plain callers keep the exact legacy call shape.
            fts_kwargs = (
                {"fts_match_query": fts_match_query}
                if fts_match_query is not None
                else {}
            )
            if id_allowlist is not None:
                fts_kwargs["id_allowlist"] = id_allowlist
            results = db.search_notes(
                search_term=search_term, limit=limit, **fts_kwargs
            )

            # Log metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_search_notes_duration",
                duration,
                labels={"status": "success"},
            )
            log_histogram("notes_library_search_notes_results_count", len(results))
            log_counter(
                "notes_library_search_notes_success",
                labels={"results_count": str(len(results))},
            )

            return results
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram(
                "notes_library_search_notes_duration",
                duration,
                labels={"status": "error"},
            )
            log_counter(
                "notes_library_search_notes_error",
                labels={"error_type": type(e).__name__},
            )
            raise

    # --- Note-Keyword Linking Methods ---
    # These methods operate on global keywords but link them to notes.
    # The user_id context from _get_db() isn't directly used for filtering these links currently
    # by ChaChaNotes_DB itself, but it sets the client_id if the link operations were to log to sync_log via self.client_id.

    def link_note_to_keyword(self, user_id: str, note_id: str, keyword_id: int) -> bool:
        db = self._get_db(user_id)
        return db.link_note_to_keyword(note_id=note_id, keyword_id=keyword_id)

    def unlink_note_from_keyword(
        self, user_id: str, note_id: str, keyword_id: int
    ) -> bool:
        db = self._get_db(user_id)
        return db.unlink_note_from_keyword(note_id=note_id, keyword_id=keyword_id)

    @staticmethod
    def _ensure_note_link_schema(db: CharactersRAGDB) -> None:
        conn = db.get_connection()
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS LocalNoteGraphLinks (
                edge_id TEXT PRIMARY KEY,
                from_note_id TEXT NOT NULL,
                to_note_id TEXT NOT NULL,
                directed INTEGER NOT NULL DEFAULT 0,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                client_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (from_note_id) REFERENCES notes(id) ON DELETE CASCADE,
                FOREIGN KEY (to_note_id) REFERENCES notes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_local_note_graph_links_from
            ON LocalNoteGraphLinks(client_id, from_note_id, deleted);

            CREATE INDEX IF NOT EXISTS idx_local_note_graph_links_to
            ON LocalNoteGraphLinks(client_id, to_note_id, deleted);
            """
        )
        conn.commit()

    @staticmethod
    def _note_link_record(row: Any) -> Dict[str, Any]:
        metadata = json.loads(row["metadata_json"] or "{}")
        return {
            "id": row["edge_id"],
            "source": row["from_note_id"],
            "target": row["to_note_id"],
            "type": "manual",
            "directed": bool(row["directed"]),
            "weight": float(row["weight"]),
            "metadata": metadata,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_note_link(
        self,
        user_id: str,
        note_id: str,
        to_note_id: str,
        *,
        directed: bool = False,
        weight: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        db = self._get_db(user_id)
        self._ensure_note_link_schema(db)
        if not db.get_note_by_id(note_id=note_id):
            raise ValueError(f"Source note '{note_id}' not found.")
        if not db.get_note_by_id(note_id=to_note_id):
            raise ValueError(f"Target note '{to_note_id}' not found.")
        edge_id = f"local:manual:{uuid.uuid4()}"
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO LocalNoteGraphLinks
                    (edge_id, from_note_id, to_note_id, directed, weight, metadata_json, client_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_id,
                    note_id,
                    to_note_id,
                    1 if directed else 0,
                    1.0 if weight is None else float(weight),
                    metadata_json,
                    user_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM LocalNoteGraphLinks WHERE edge_id = ?",
                (edge_id,),
            ).fetchone()
        return self._note_link_record(row)

    def list_note_links(
        self,
        user_id: str,
        *,
        center_note_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        self._ensure_note_link_schema(db)
        params: list[Any] = [user_id]
        where = "client_id = ? AND deleted = 0"
        if center_note_id:
            where += " AND (from_note_id = ? OR to_note_id = ?)"
            params.extend([center_note_id, center_note_id])
        params.append(max(0, int(limit)))
        rows = (
            db.get_connection()
            .execute(
                f"""
            SELECT *
            FROM LocalNoteGraphLinks
            WHERE {where}
            ORDER BY created_at ASC, edge_id ASC
            LIMIT ?
            """,
                tuple(params),
            )
            .fetchall()
        )
        return [self._note_link_record(row) for row in rows]

    def delete_note_link(self, user_id: str, edge_id: str) -> Dict[str, Any]:
        db = self._get_db(user_id)
        self._ensure_note_link_schema(db)
        with db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE LocalNoteGraphLinks
                SET deleted = 1, updated_at = CURRENT_TIMESTAMP
                WHERE client_id = ? AND edge_id = ? AND deleted = 0
                """,
                (user_id, edge_id),
            )
        return {"deleted": cursor.rowcount > 0, "edge_id": edge_id}

    def get_keywords_for_note(self, user_id: str, note_id: str) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        return [
            row
            for row in db.get_keywords_for_note(note_id=note_id)
            if not _is_internal_research_keyword(row)
        ]

    def get_notes_for_keyword(
        self, user_id: str, keyword_id: int, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        keyword = db.get_keyword_by_id(keyword_id=keyword_id)
        if _is_internal_research_keyword(keyword):
            return []
        return db.get_notes_for_keyword(
            keyword_id=keyword_id, limit=limit, offset=offset
        )

    # --- Keyword Methods (Keywords are global in ChaChaNotes_DB) ---
    # The user_id is passed to _get_db to maintain consistency, but keywords are global.
    # The client_id set by _get_db() when adding/deleting keywords will be the user_id.

    def add_keyword(self, user_id: str, keyword_text: str) -> Optional[int]:
        db = self._get_db(user_id)
        return db.add_keyword(keyword_text=keyword_text)

    def get_keyword_by_id(
        self, user_id: str, keyword_id: int
    ) -> Optional[Dict[str, Any]]:
        db = self._get_db(user_id)
        row = db.get_keyword_by_id(keyword_id=keyword_id)
        return None if _is_internal_research_keyword(row) else row

    def get_keyword_by_text(
        self, user_id: str, keyword_text: str
    ) -> Optional[Dict[str, Any]]:
        db = self._get_db(user_id)
        row = db.get_keyword_by_text(keyword_text=keyword_text)
        return None if _is_internal_research_keyword(row) else row

    def list_keywords(
        self, user_id: str, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        return [
            row
            for row in db.list_keywords(limit=limit, offset=offset)
            if not _is_internal_research_keyword(row)
        ]

    def soft_delete_keyword(
        self, user_id: str, keyword_id: int, expected_version: int
    ) -> bool:
        db = self._get_db(user_id)
        return db.soft_delete_keyword(
            keyword_id=keyword_id, expected_version=expected_version
        )

    def search_keywords(
        self, user_id: str, search_term: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        return [
            row
            for row in db.search_keywords(search_term=search_term, limit=limit)
            if not _is_internal_research_keyword(row)
        ]

    # --- Character Card Methods ---

    def add_character_card(
        self, user_id: str, character_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Adds a new character card for the specified user.
        Assumes user_id is used by the underlying DB method if it needs it for multi-user contexts,
        or is ignored if the DB is single-user context for characters.
        """
        # The _get_db method might not be appropriate if self.unified_db is always used.
        # Directly use self.unified_db if character operations are always on the global DB.
        if not self.unified_db:
            logger.error("Unified database not available in add_character_card.")
            raise CharactersRAGDBError("Unified database not available.")
        logger.debug(
            f"Service: Adding character for user '{user_id}' with data: {character_data.get('name')}"
        )
        # ChaChaNotes_DB.add_character_card expects user_id as a named argument.
        return self.unified_db.add_character_card(
            character_data=character_data, user_id=user_id
        )

    def update_character_card(
        self,
        character_id: str,
        user_id: str,
        update_data: Dict[str, Any],
        expected_version: Optional[int],
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing character card for the specified user with optimistic locking."""
        if not self.unified_db:
            logger.error("Unified database not available in update_character_card.")
            raise CharactersRAGDBError("Unified database not available.")
        logger.debug(
            f"Service: Updating character ID '{character_id}' for user '{user_id}'. Version: {expected_version}"
        )
        # ChaChaNotes_DB.update_character_card expects user_id.
        return self.unified_db.update_character_card(
            character_id=character_id,
            user_id=user_id,
            update_data=update_data,
            expected_version=expected_version,
        )

    # --- Resource Management ---

    def close_all_user_connections(self):
        with self._db_lock:
            logger.info(
                f"Closing all {len(self._db_instances)} cached user-context DB instances."
            )
            for user_id, db_instance in self._db_instances.items():
                try:
                    # Each db_instance is a CharactersRAGDB, call its close_connection
                    db_instance.close_connection()
                    logger.debug(f"Closed DB instance for user context '{user_id}'.")
                except Exception as e:
                    logger.error(
                        f"Error closing DB instance for user context '{user_id}': {e}",
                        exc_info=True,
                    )
            self._db_instances.clear()
        logger.info(
            "All cached user-context DB instances have been processed for closure."
        )

    def close_user_connection(self, user_id: str):
        with self._db_lock:
            if user_id in self._db_instances:
                db_instance = self._db_instances.pop(user_id)
                try:
                    db_instance.close_connection()
                    logger.info(
                        f"Closed and removed DB instance for user context '{user_id}'."
                    )
                except Exception as e:
                    logger.error(
                        f"Error closing DB instance for user context '{user_id}': {e}",
                        exc_info=True,
                    )
            else:
                logger.debug(
                    f"No active DB instance found in cache for user context '{user_id}' to close."
                )


#
# End of Notes_Library.py
#######################################################################################################################
