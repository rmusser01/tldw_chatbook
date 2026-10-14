# keyword_store.py (TASK-34903)
#########################################
# Shared keyword write-path store for the Media and Prompts databases.
#
# Media (``Client_Media_DB_v2.py``) and Prompts (``Prompts_DB.py``) carried two
# near-identical copies of the keyword write path (add/undelete, link/unlink on
# the entity junction table, soft delete, FTS upsert/delete). Per ADR-177 the
# consolidation is a HELPER the two hosts delegate to -- parameterized by a
# ``KeywordStoreSpec`` for their real differences (table names, normalization,
# exception construction, link-deletion scope) -- and deliberately NOT a base
# class. Public host APIs are unchanged: the host methods are thin delegates.
#
# Notes on deliberate unification choices:
#
# - ``update_fts`` scrubs NUL bytes (``keyword.replace("\\x00", " ")``) before
#   writing the FTS row. That scrub existed only in Media's copy; it now also
#   covers the Prompts keyword FTS upsert, where a NUL-bearing keyword
#   previously corrupted the FTS index. Strict improvement, no behavior lost.
# - ``make_database_error`` is called with the canonical (Media) message text.
#   Hosts whose original wording diverged (Prompts) rewrite the message inside
#   their spec callable, so each host keeps raising its own exception class
#   with its own historical message.
# - sqlite3 failures are re-raised ``from e`` everywhere; Prompts historically
#   used ``from None`` at two wrap sites, so those now carry ``__cause__``
#   (strictly more debuggable, no test observes the difference).
# - The host's own exception classes are unreachable from this module by
#   design: every raise goes through a spec callable, so there is no import
#   edge back into either host module.
# - ``get_active_keyword_by_text`` catches ``Exception`` (returning None)
#   where Prompts' original caught ``(DatabaseError, sqlite3.Error)``;
#   foreign errors now surface as None on this read helper.
# - Consolidated logging means Prompts now logs an error line for a
#   ConflictError its original let pass silently, and Prompts' not-found /
#   soft-delete log wording uses the shared (Media) texts.
####

import logging
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

#
########################################################################################################################


@dataclass(frozen=True)
class KeywordStoreSpec:
    """Per-host parameterization of the shared keyword write path.

    Attributes:
        keywords_table: Keyword table name (``Keywords`` / ``PromptKeywordsTable``).
        fts_table: Keyword FTS table name (``keyword_fts`` / ``prompt_keywords_fts``).
        link_table: Entity<->keyword junction table (``MediaKeywords`` /
            ``PromptKeywordLinks``).
        entity_table: Parent entity table (``Media`` / ``Prompts``).
        entity_fk: Junction-table FK column for the parent (``media_id`` /
            ``prompt_id``).
        entity_label: Human label for error messages (``Media`` / ``Prompt``).
        sync_entity: sync_log entity name for keyword-row events.
        link_sync_entity: sync_log entity name for link/unlink events.
        entity_uuid_key: Payload key for the parent UUID in link sync events
            (``media_uuid`` / ``prompt_uuid``).
        normalize_keyword: Single-keyword normalization (storage form).
        canonicalize_keywords: List normalization/validation for membership
            updates; returns the sorted unique target set.
        make_input_error: ``str -> Exception`` constructing the host's
            InputError.
        make_database_error: ``str -> Exception`` constructing the host's
            DatabaseError. Receives the canonical (Media) message; hosts with
            divergent historical wording rewrite it here.
        make_conflict: ``(site, kw_id) -> Exception`` constructing the host's
            ConflictError; ``site`` is ``"undelete"`` or ``"soft_delete"``.
        unlink_all_links: Soft-delete link removal scope. ``False``: remove
            only links to ACTIVE parents (IN-list). ``True``: remove ALL links
            for the keyword.
    """

    keywords_table: str
    fts_table: str
    link_table: str
    entity_table: str
    entity_fk: str
    entity_label: str
    sync_entity: str
    link_sync_entity: str
    entity_uuid_key: str
    normalize_keyword: Callable[[str], str]
    canonicalize_keywords: Callable[[List[str]], List[str]]
    make_input_error: Callable[[str], Exception]
    make_database_error: Callable[[str], Exception]
    make_conflict: Callable[[str, int], Exception]
    unlink_all_links: bool


class KeywordStore:
    """The keyword write path shared by the Media and Prompts databases.

    The host (``db``) is duck-typed; it must provide ``transaction()``,
    ``_log_sync_event(conn, entity, uuid, op, version, payload)``,
    ``_generate_uuid()``, ``_get_current_utc_timestamp_str()``,
    ``execute_query()``, and a ``client_id`` attribute -- exactly the surface
    the pre-store host methods used on ``self``.
    """

    def __init__(self, db, spec: KeywordStoreSpec):
        """
        Args:
            db: The host database instance (duck-typed, see class docstring).
            spec: Per-host parameterization of table names and callables.
        """
        self.db = db
        self.spec = spec

    # --- Keyword row lifecycle ---

    def add_keyword_full(
        self, keyword_text: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """Adds a new keyword or undeletes an existing soft-deleted one.

        Ported verbatim (modulo table names) from the twin bodies of Media
        ``add_keyword`` / Prompts ``_add_keyword_full``: SELECT by keyword;
        undelete branch (optimistic-lock UPDATE, payload re-fetch, "update"
        sync event, FTS upsert); already-active no-op return; insert branch
        (uuid generation, lastrowid check, payload re-fetch, "create" sync
        event, FTS upsert).

        Args:
            keyword_text (str): The keyword text to add or activate.

        Returns:
            Tuple[Optional[int], Optional[str]]: The keyword's database ID and
                UUID. Returns (None, None) or raises on failure.

        Raises:
            Exception: The host's InputError if the keyword is empty; the
                host's ConflictError if an undelete loses a version race; the
                host's DatabaseError for other database errors.
        """
        if not keyword_text or not keyword_text.strip():
            raise self.spec.make_input_error("Keyword cannot be empty.")
        keyword = self.spec.normalize_keyword(keyword_text)
        current_time = self.db._get_current_utc_timestamp_str()  # Get current time once
        client_id = self.db.client_id

        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT id, uuid, deleted, version FROM {self.spec.keywords_table} WHERE keyword = ?",
                    (keyword,),
                )
                existing = cursor.fetchone()

                if existing:
                    kw_id, kw_uuid, is_deleted, current_version = (
                        existing["id"],
                        existing["uuid"],
                        existing["deleted"],
                        existing["version"],
                    )
                    if is_deleted:
                        new_version = current_version + 1
                        logger.info(
                            f"Undeleting keyword '{keyword}' (ID: {kw_id}). New ver: {new_version}"
                        )
                        # Pass current_time for last_modified
                        cursor.execute(
                            f"UPDATE {self.spec.keywords_table} SET deleted=0, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                            (
                                current_time,
                                new_version,
                                client_id,
                                kw_id,
                                current_version,
                            ),
                        )
                        if cursor.rowcount == 0:
                            raise self.spec.make_conflict("undelete", kw_id)

                        # Fetch data for payload AFTER update to get correct last_modified
                        cursor.execute(
                            f"SELECT * FROM {self.spec.keywords_table} WHERE id=?",
                            (kw_id,),
                        )
                        payload_data = dict(cursor.fetchone())
                        self.db._log_sync_event(
                            conn,
                            self.spec.sync_entity,
                            kw_uuid,
                            "update",
                            new_version,
                            payload_data,
                        )
                        self.update_fts(conn, kw_id, keyword)
                        return kw_id, kw_uuid
                    else:
                        logger.debug(f"Keyword '{keyword}' already active.")
                        return kw_id, kw_uuid
                else:
                    new_uuid = self.db._generate_uuid()
                    new_version = 1
                    logger.info(f"Adding new keyword '{keyword}' UUID {new_uuid}")
                    # Pass current_time for last_modified
                    cursor.execute(
                        f"INSERT INTO {self.spec.keywords_table} (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, 0)",
                        (keyword, new_uuid, current_time, new_version, client_id),
                    )
                    kw_id = cursor.lastrowid
                    if not kw_id:
                        raise self.spec.make_database_error(
                            "Failed to get last row ID for new keyword."
                        )

                    # Fetch data for payload AFTER insert to get correct last_modified
                    cursor.execute(
                        f"SELECT * FROM {self.spec.keywords_table} WHERE id=?",
                        (kw_id,),
                    )
                    payload_data = dict(cursor.fetchone())
                    self.db._log_sync_event(
                        conn,
                        self.spec.sync_entity,
                        new_uuid,
                        "create",
                        new_version,
                        payload_data,
                    )
                    self.update_fts(conn, kw_id, keyword)
                    return kw_id, new_uuid
        except Exception as e:
            logger.opt(exception=True).error(
                f"Error in add_keyword for '{keyword}': {e}"
            )
            if isinstance(e, sqlite3.Error):
                raise self.spec.make_database_error(
                    f"Failed to add/update keyword: {e}"
                ) from e
            raise

    def soft_delete_keyword(self, keyword_text: str) -> bool:
        """Soft deletes a keyword by text and unlinks it from active parents.

        Ported from the twin bodies of Media/Prompts ``soft_delete_keyword``.
        The empty-keyword InputError is raised at the HOST call site (the two
        hosts word it differently); this method receives raw text and
        normalizes via the spec.

        Args:
            keyword_text (str): The keyword text to soft delete (case-insensitive).

        Returns:
            bool: True if soft-deleted; False if not found or already deleted.

        Raises:
            Exception: The host's ConflictError on a version race; the host's
                DatabaseError for other database errors.
        """
        keyword = self.spec.normalize_keyword(keyword_text)
        current_time = self.db._get_current_utc_timestamp_str()  # Get time
        client_id = self.db.client_id

        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT id, uuid, version FROM {self.spec.keywords_table} WHERE keyword = ? AND deleted = 0",
                    (keyword,),
                )
                keyword_info = cursor.fetchone()
                if not keyword_info:
                    logger.warning(f"Keyword '{keyword}' not found/deleted.")
                    return False
                keyword_id, keyword_uuid, current_version = (
                    keyword_info["id"],
                    keyword_info["uuid"],
                    keyword_info["version"],
                )
                new_version = current_version + 1

                logger.info(
                    f"Soft deleting keyword '{keyword}' (ID: {keyword_id}). New ver: {new_version}"
                )
                # Pass current_time for last_modified
                cursor.execute(
                    f"UPDATE {self.spec.keywords_table} SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                    (current_time, new_version, client_id, keyword_id, current_version),
                )
                if cursor.rowcount == 0:
                    raise self.spec.make_conflict("soft_delete", keyword_id)

                # Payload reflects the state *after* the update
                delete_payload = {
                    "uuid": keyword_uuid,
                    "last_modified": current_time,
                    "version": new_version,
                    "client_id": client_id,
                    "deleted": 1,
                }
                self.db._log_sync_event(
                    conn,
                    self.spec.sync_entity,
                    keyword_uuid,
                    "delete",
                    new_version,
                    delete_payload,
                )
                self.delete_fts(conn, keyword_id)

                # Unlinking logic remains the same
                cursor.execute(
                    f"SELECT lk.{self.spec.entity_fk} AS entity_id, e.uuid AS entity_uuid FROM {self.spec.link_table} lk JOIN {self.spec.entity_table} e ON lk.{self.spec.entity_fk} = e.id WHERE lk.keyword_id = ? AND e.deleted = 0",
                    (keyword_id,),
                )
                linked_entities = cursor.fetchall()
                if linked_entities:
                    if self.spec.unlink_all_links:
                        cursor.execute(
                            f"DELETE FROM {self.spec.link_table} WHERE keyword_id = ?",
                            (keyword_id,),
                        )
                    else:
                        entity_ids = [m["entity_id"] for m in linked_entities]
                        placeholders = ",".join("?" * len(entity_ids))
                        cursor.execute(
                            f"DELETE FROM {self.spec.link_table} WHERE keyword_id = ? AND {self.spec.entity_fk} IN ({placeholders})",
                            (keyword_id, *entity_ids),
                        )
                    unlink_version = 1
                    deleted_link_count = (
                        cursor.rowcount
                    )  # Get actual count of deleted links
                    for entity_link in linked_entities:
                        link_uuid = f"{entity_link['entity_uuid']}_{keyword_uuid}"
                        unlink_payload = {
                            self.spec.entity_uuid_key: entity_link["entity_uuid"],
                            "keyword_uuid": keyword_uuid,
                        }
                        self.db._log_sync_event(
                            conn,
                            self.spec.link_sync_entity,
                            link_uuid,
                            "unlink",
                            unlink_version,
                            unlink_payload,
                        )
                    logger.info(
                        f"Unlinked keyword '{keyword}' from {deleted_link_count} items."
                    )
            return True
        except Exception as e:
            logger.opt(exception=True).error(
                f"Error soft delete keyword '{keyword}': {e}"
            )
            if isinstance(e, sqlite3.Error):
                raise self.spec.make_database_error(
                    f"Failed soft delete keyword: {e}"
                ) from e
            raise

    # --- FTS maintenance ---

    def update_fts(
        self, conn: sqlite3.Connection, keyword_id: int, keyword: str
    ) -> None:
        """Updates or inserts the keyword's FTS row (Media's canonical body).

        Uses INSERT OR REPLACE based on the keyword rowid. Should be called
        within a transaction after keyword insert/update/undelete.

        Args:
            conn (sqlite3.Connection): The database connection (within transaction).
            keyword_id (int): The ID (rowid) of the keyword row.
            keyword (str): The keyword text.

        Raises:
            Exception: The host's DatabaseError if the FTS update fails.
        """
        # Remove null bytes from keyword as they break FTS indexing
        keyword = keyword.replace("\x00", " ")
        try:
            # Use INSERT OR REPLACE
            conn.execute(
                f"INSERT OR REPLACE INTO {self.spec.fts_table} (rowid, keyword) VALUES (?, ?)",
                (keyword_id, keyword),
            )
            logging.debug(
                f"Updated FTS (insert or replace) for Keyword ID {keyword_id}"
            )
        except sqlite3.Error as e:
            logging.error(
                f"Failed to update keyword_fts for Keyword ID {keyword_id}: {e}",
                exc_info=True,
            )
            raise self.spec.make_database_error(
                f"Failed to update FTS for Keyword ID {keyword_id}: {e}"
            ) from e

    def delete_fts(self, conn: sqlite3.Connection, keyword_id: int) -> None:
        """Deletes the keyword's FTS row (Media's canonical body).

        Deletes the FTS entry corresponding to the given keyword rowid.
        Should be called within a transaction after keyword soft delete.
        Ignores if the entry doesn't exist.

        Args:
            conn (sqlite3.Connection): The database connection (within transaction).
            keyword_id (int): The ID (rowid) of the keyword row.

        Raises:
            Exception: The host's DatabaseError if the FTS deletion fails.
        """
        try:
            conn.execute(
                f"DELETE FROM {self.spec.fts_table} WHERE rowid = ?", (keyword_id,)
            )
            logging.debug(f"Deleted FTS entry for Keyword ID {keyword_id}")
        except sqlite3.Error as e:
            logging.error(
                f"Failed to delete from keyword_fts for Keyword ID {keyword_id}: {e}",
                exc_info=True,
            )
            raise self.spec.make_database_error(
                f"Failed to delete FTS for Keyword ID {keyword_id}: {e}"
            ) from e

    # --- Entity keyword membership ---

    def update_keywords_for_entity(self, entity_id: int, keywords: List[str]) -> bool:
        """Synchronizes the keywords linked to a parent Media/Prompt row.

        Ported from the twin bodies of Media ``update_keywords_for_media`` /
        Prompts ``update_keywords_for_prompt``: parent-active check, current
        active links, per-new-keyword ``add_keyword_full``, IN-list link
        removals, INSERT OR IGNORE link adds, and per-link unlink/link sync
        events with composite ``f"{entity_uuid}_{keyword_uuid}"`` identifiers.

        Args:
            entity_id (int): The ID of the parent row whose keywords to update.
            keywords (List[str]): The desired keyword list; empty removes all.

        Returns:
            bool: True if the operation completed (even with no changes).

        Raises:
            Exception: The host's InputError if the parent is missing/deleted
                or the keyword list is invalid; the host's ConflictError if an
                undelete loses a version race; the host's DatabaseError for
                other database errors.
        """
        entity_label = self.spec.entity_label.lower()
        valid_keywords = self.spec.canonicalize_keywords(keywords)
        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT uuid FROM {self.spec.entity_table} WHERE id = ? AND deleted = 0",
                    (entity_id,),
                )  # Check parent active
                entity_info = cursor.fetchone()
                if not entity_info:
                    raise self.spec.make_input_error(
                        f"Cannot update keywords: {self.spec.entity_label} ID {entity_id} not found or deleted."
                    )
                entity_uuid = entity_info["uuid"]

                cursor.execute(
                    f"SELECT lk.keyword_id, k.uuid AS keyword_uuid FROM {self.spec.link_table} lk JOIN {self.spec.keywords_table} k ON lk.keyword_id = k.id WHERE lk.{self.spec.entity_fk} = ? AND k.deleted = 0",
                    (entity_id,),
                )
                current_links = {
                    row["keyword_id"]: row["keyword_uuid"] for row in cursor.fetchall()
                }
                current_keyword_ids = set(current_links.keys())

                target_keyword_data = {}
                if valid_keywords:
                    for kw_text in valid_keywords:
                        kw_id, kw_uuid = self.add_keyword_full(
                            kw_text
                        )  # Handles create/undelete/logging/FTS for Keywords
                        if kw_id and kw_uuid:
                            target_keyword_data[kw_id] = kw_uuid
                        else:
                            raise self.spec.make_database_error(
                                f"Failed get/add keyword '{kw_text}'"
                            )

                target_keyword_ids = set(target_keyword_data.keys())
                ids_to_add = target_keyword_ids - current_keyword_ids
                ids_to_remove = current_keyword_ids - target_keyword_ids
                link_sync_version = 1

                if ids_to_remove:
                    remove_placeholders = ",".join("?" * len(ids_to_remove))
                    cursor.execute(
                        f"DELETE FROM {self.spec.link_table} WHERE {self.spec.entity_fk} = ? AND keyword_id IN ({remove_placeholders})",
                        (entity_id, *list(ids_to_remove)),
                    )
                    for removed_id in ids_to_remove:
                        keyword_uuid = current_links.get(removed_id)
                        if keyword_uuid:
                            link_uuid = f"{entity_uuid}_{keyword_uuid}"
                            payload = {
                                self.spec.entity_uuid_key: entity_uuid,
                                "keyword_uuid": keyword_uuid,
                            }
                            self.db._log_sync_event(
                                conn,
                                self.spec.link_sync_entity,
                                link_uuid,
                                "unlink",
                                link_sync_version,
                                payload,
                            )

                if ids_to_add:
                    insert_params = [(entity_id, kid) for kid in ids_to_add]
                    cursor.executemany(
                        f"INSERT OR IGNORE INTO {self.spec.link_table} ({self.spec.entity_fk}, keyword_id) VALUES (?, ?)",
                        insert_params,
                    )
                    # Log links - Note: IGNORE means we might log links that weren't actually inserted if race condition. Robust check is complex.
                    for added_id in ids_to_add:
                        keyword_uuid = target_keyword_data.get(added_id)
                        if keyword_uuid:
                            link_uuid = f"{entity_uuid}_{keyword_uuid}"
                            payload = {
                                self.spec.entity_uuid_key: entity_uuid,
                                "keyword_uuid": keyword_uuid,
                            }
                            self.db._log_sync_event(
                                conn,
                                self.spec.link_sync_entity,
                                link_uuid,
                                "link",
                                link_sync_version,
                                payload,
                            )

                if ids_to_add or ids_to_remove:
                    logger.debug(
                        f"Keywords updated {entity_label} {entity_id}. Added: {len(ids_to_add)}, Removed: {len(ids_to_remove)}."
                    )
                else:
                    logger.debug(f"No keyword changes {entity_label} {entity_id}.")
                return True
        except Exception as e:
            logger.opt(exception=True).error(
                f"Error updating keywords {entity_label} {entity_id}: {e}"
            )
            if isinstance(e, sqlite3.Error):
                raise self.spec.make_database_error(
                    f"Keyword update failed: {e}"
                ) from e
            raise

    # --- Host-specific read helpers ---

    def usage_stats(self) -> List[Dict[str, Any]]:
        """Usage statistics for all active keywords (Media host only).

        Returns:
            List[Dict[str, Any]]: Per-keyword dicts with id, keyword, uuid,
                last_modified, and usage_count (distinct linked parents).

        Raises:
            Exception: The host's DatabaseError for database operation errors.
        """
        try:
            query = f"""
                SELECT
                    k.id,
                    k.keyword,
                    k.uuid,
                    k.last_modified,
                    COUNT(DISTINCT lk.{self.spec.entity_fk}) as usage_count
                FROM {self.spec.keywords_table} k
                LEFT JOIN {self.spec.link_table} lk ON k.id = lk.keyword_id
                WHERE k.deleted = 0
                GROUP BY k.id, k.keyword, k.uuid, k.last_modified
                ORDER BY usage_count DESC, k.keyword COLLATE NOCASE
            """

            cursor = self.db.execute_query(query)
            results = [dict(row) for row in cursor.fetchall()]

            logger.debug(f"Retrieved usage stats for {len(results)} keywords")
            return results

        except sqlite3.Error as e:
            logger.opt(exception=True).error(
                f"SQLite error getting keyword usage stats: {e}"
            )
            raise self.spec.make_database_error(
                f"Failed to get keyword usage stats: {e}"
            ) from e
        except Exception as e:
            logger.opt(exception=True).error(
                f"Unexpected error getting keyword usage stats: {e}"
            )
            raise self.spec.make_database_error(f"Unexpected stats error: {e}") from e

    def get_active_keyword_by_text(self, keyword_text: str) -> Optional[Dict[str, Any]]:
        """Fetches an active (not deleted) keyword by exact normalized text.

        Prompts host only.

        Args:
            keyword_text (str): The keyword text to search for.

        Returns:
            Optional[Dict[str, Any]]: The keyword's data if found and active,
                else None (also None on query error, as the original did).
        """
        if not keyword_text or not keyword_text.strip():
            return None  # Or raise InputError if strictness is preferred here
        normalized_keyword = self.spec.normalize_keyword(keyword_text)
        query = f"SELECT id, uuid, keyword, last_modified, version, client_id FROM {self.spec.keywords_table} WHERE keyword = ? AND deleted = 0"
        try:
            cursor = self.db.execute_query(query, (normalized_keyword,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except Exception as e:
            logger.error(
                f"Error fetching active keyword by text '{normalized_keyword}': {e}"
            )
            # Depending on desired strictness, could raise or return None
            # For a simple check, returning None on error is acceptable if the next step handles it.
            return None
