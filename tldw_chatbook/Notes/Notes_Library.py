# Notes_Library.py
# Description: This module provides a service layer for managing notes and note keywords
#
# Imports
import logging
import threading
import sqlite3  # For exception handling in _get_db
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
#
# Third-Party Imports
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
    InputError,
    ConflictError
)
from tldw_chatbook.config import chachanotes_db as global_db_from_config
from ..Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:

logger = logging.getLogger(__name__)


class NotesInteropService:
    def __init__(self,
                 base_db_directory: Union[str, Path],
                 api_client_id: str, # This api_client_id might be a fallback or general app id
                 global_db_to_use: Optional[CharactersRAGDB] = None):

        self.base_db_directory = Path(base_db_directory).resolve()
        # self.api_client_id is not directly used if _get_db uses user_id as client_id
        # It's good to have it for context or if some methods need a generic app client_id.
        self.api_client_id = api_client_id

        self._db_instances: Dict[str, CharactersRAGDB] = {} # Cache instances per user_id (as client_id)
        self._db_lock = threading.Lock()

        try:
            self.base_db_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"NotesInteropService: Ensured base directory exists: {self.base_db_directory}")
        except OSError as e:
            logger.error(f"Failed to create base DB directory {self.base_db_directory}: {e}")
            raise CharactersRAGDBError(f"Failed to create base DB directory {self.base_db_directory}: {e}") from e

        # Store the global DB instance
        if global_db_to_use:
            self.unified_db_template = global_db_to_use # Store the template instance
            logger.info(f"NotesInteropService initialized with unified DB template: {self.unified_db_template.db_path_str}")
        elif global_db_from_config:
            self.unified_db_template = global_db_from_config
            logger.info(f"NotesInteropService using imported global config DB template: {self.unified_db_template.db_path_str}")
        else:
            self.unified_db_template = None

        if not self.unified_db_template:
            logger.critical("NotesInteropService CRITICAL: No unified database template instance available!")
            raise CharactersRAGDBError("No unified database template for NotesInteropService.")

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
            if user_id in self._db_instances: # Double-check
                return self._db_instances[user_id]

            if not self.unified_db_template:
                logger.critical("NotesInteropService: Unified database template (self.unified_db_template) is not initialized!")
                raise CharactersRAGDBError("Unified database template is not available in NotesInteropService.")

            # Create a new CharactersRAGDB instance for this user_id,
            # ensuring it points to the *global unified database file path*
            # and uses the current `user_id` as its `client_id`.
            try:
                unified_db_file_path = self.unified_db_template.db_path_str
                logger.info(f"Creating new CharactersRAGDB instance for context of user '{user_id}'. "
                            f"DB File: {unified_db_file_path}, Client ID for ops: '{user_id}'.")

                db_instance = CharactersRAGDB(
                    db_path=unified_db_file_path, # Use the path from the unified DB template
                    client_id=user_id             # Use the passed user_id as the client_id for this instance
                )
                self._db_instances[user_id] = db_instance # Cache it
                logger.info(f"Successfully initialized dynamic CharactersRAGDB instance for user context '{user_id}'.")
                return db_instance
            except (CharactersRAGDBError, SchemaError, sqlite3.Error) as e:
                logger.error(f"Failed to initialize dynamic CharactersRAGDB instance for user '{user_id}': {e}", exc_info=True)
                raise
            except Exception as e: # Catch any other unexpected Python error
                logger.error(f"Unexpected error initializing dynamic CharactersRAGDB for user '{user_id}': {e}", exc_info=True)
                raise CharactersRAGDBError(f"Unexpected error initializing DB instance for user {user_id}: {e}") from e

    # --- Note Methods ---

    def add_note(self, user_id: str, title: str, content: str, note_id: Optional[str] = None) -> str:
        """
        Adds a new note for the specified user. The user_id will be used as the client_id.
        """
        start_time = time.time()
        log_counter("notes_library_add_note_attempt", labels={"has_note_id": str(note_id is not None)})
        
        try:
            db = self._get_db(user_id)
            created_note_id = db.add_note(title=title, content=content, note_id=note_id)
            if created_note_id is None:
                logger.error(f"add_note for user_id '{user_id}' (as client_id) returned None unexpectedly for title '{title}'.")
                log_counter("notes_library_add_note_error", labels={"error_type": "null_id_returned"})
                raise CharactersRAGDBError("Failed to create note, received None ID unexpectedly.")
            
            # Log success metrics
            duration = time.time() - start_time
            log_histogram("notes_library_add_note_duration", duration, labels={"status": "success"})
            log_histogram("notes_library_note_content_length", len(content))
            log_counter("notes_library_add_note_success")
            
            return created_note_id
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("notes_library_add_note_duration", duration, labels={"status": "error"})
            log_counter("notes_library_add_note_error", labels={"error_type": type(e).__name__})
            raise

    def get_note_by_id(self, user_id: str, note_id: str) -> Optional[Dict[str, Any]]:
        start_time = time.time()
        log_counter("notes_library_get_note_attempt")
        
        db = self._get_db(user_id) # user_id here is mainly for consistency or if _get_db has other uses
        result = db.get_note_by_id(note_id=note_id) # The actual filtering by user would be in SQL if notes were user-specific
        
        # Log metrics
        duration = time.time() - start_time
        log_histogram("notes_library_get_note_duration", duration)
        log_counter("notes_library_get_note_result", labels={"found": str(result is not None)})
        
        return result

    def list_notes(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        start_time = time.time()
        log_counter("notes_library_list_notes_attempt", labels={"limit": str(limit), "has_offset": str(offset > 0)})
        
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
        log_counter("notes_library_list_notes_success", labels={"count": str(len(results))})
        
        return results

    def update_note(self, user_id: str, note_id: str, update_data: Dict[str, Any], expected_version: int) -> bool:
        start_time = time.time()
        log_counter("notes_library_update_note_attempt", labels={
            "fields_count": str(len(update_data)),
            "has_title": str('title' in update_data),
            "has_content": str('content' in update_data)
        })
        
        try:
            db = self._get_db(user_id) # The db instance will have user_id as its client_id for the update
            result = db.update_note(note_id=note_id, update_data=update_data, expected_version=expected_version)
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("notes_library_update_note_duration", duration, labels={"status": "success" if result else "conflict"})
            log_counter("notes_library_update_note_result", labels={"success": str(result)})
            
            return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("notes_library_update_note_duration", duration, labels={"status": "error"})
            log_counter("notes_library_update_note_error", labels={"error_type": type(e).__name__})
            raise

    def soft_delete_note(self, user_id: str, note_id: str, expected_version: int) -> bool:
        start_time = time.time()
        log_counter("notes_library_delete_note_attempt")
        
        try:
            db = self._get_db(user_id) # client_id for operation comes from db instance
            result = db.soft_delete_note(note_id=note_id, expected_version=expected_version)
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("notes_library_delete_note_duration", duration, labels={"status": "success" if result else "conflict"})
            log_counter("notes_library_delete_note_result", labels={"success": str(result)})
            
            return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("notes_library_delete_note_duration", duration, labels={"status": "error"})
            log_counter("notes_library_delete_note_error", labels={"error_type": type(e).__name__})
            raise

    def search_notes(self, user_id: str, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        start_time = time.time()
        log_counter("notes_library_search_notes_attempt", labels={
            "search_term_length": str(len(search_term)),
            "limit": str(limit)
        })
        try:
            db = self._get_db(user_id)
            # Similar to list_notes, if search should be user-specific, CharactersRAGDB.search_notes needs adjustment.
            results = db.search_notes(search_term=search_term, limit=limit)
            
            # Log metrics
            duration = time.time() - start_time
            log_histogram("notes_library_search_notes_duration", duration, labels={"status": "success"})
            log_histogram("notes_library_search_notes_results_count", len(results))
            log_counter("notes_library_search_notes_success", labels={"results_count": str(len(results))})
            
            return results
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("notes_library_search_notes_duration", duration, labels={"status": "error"})
            log_counter("notes_library_search_notes_error", labels={"error_type": type(e).__name__})
            raise

    # --- Note-Keyword Linking Methods ---
    # These methods operate on global keywords but link them to notes.
    # The user_id context from _get_db() isn't directly used for filtering these links currently
    # by ChaChaNotes_DB itself, but it sets the client_id if the link operations were to log to sync_log via self.client_id.

    def link_note_to_keyword(self, user_id: str, note_id: str, keyword_id: int) -> bool:
        db = self._get_db(user_id)
        return db.link_note_to_keyword(note_id=note_id, keyword_id=keyword_id)

    def unlink_note_from_keyword(self, user_id: str, note_id: str, keyword_id: int) -> bool:
        db = self._get_db(user_id)
        return db.unlink_note_from_keyword(note_id=note_id, keyword_id=keyword_id)

    def get_keywords_for_note(self, user_id: str, note_id: str) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        return db.get_keywords_for_note(note_id=note_id)

    def get_notes_for_keyword(self, user_id: str, keyword_id: int, limit: int = 50, offset: int = 0) -> List[
        Dict[str, Any]]:
        db = self._get_db(user_id)
        return db.get_notes_for_keyword(keyword_id=keyword_id, limit=limit, offset=offset)

    # --- Keyword Methods (Keywords are global in ChaChaNotes_DB) ---
    # The user_id is passed to _get_db to maintain consistency, but keywords are global.
    # The client_id set by _get_db() when adding/deleting keywords will be the user_id.

    def add_keyword(self, user_id: str, keyword_text: str) -> Optional[int]:
        db = self._get_db(user_id)
        return db.add_keyword(keyword_text=keyword_text)

    def get_keyword_by_id(self, user_id: str, keyword_id: int) -> Optional[Dict[str, Any]]:
        db = self._get_db(user_id)
        return db.get_keyword_by_id(keyword_id=keyword_id)

    def get_keyword_by_text(self, user_id: str, keyword_text: str) -> Optional[Dict[str, Any]]:
        db = self._get_db(user_id)
        return db.get_keyword_by_text(keyword_text=keyword_text)

    def list_keywords(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        return db.list_keywords(limit=limit, offset=offset)

    def soft_delete_keyword(self, user_id: str, keyword_id: int, expected_version: int) -> bool:
        db = self._get_db(user_id)
        return db.soft_delete_keyword(keyword_id=keyword_id, expected_version=expected_version)

    def search_keywords(self, user_id: str, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        db = self._get_db(user_id)
        return db.search_keywords(search_term=search_term, limit=limit)

    # --- Character Card Methods ---

    def add_character_card(self, user_id: str, character_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a new character card for the specified user.
        Assumes user_id is used by the underlying DB method if it needs it for multi-user contexts,
        or is ignored if the DB is single-user context for characters.
        """
        # The _get_db method might not be appropriate if self.unified_db is always used.
        # Directly use self.unified_db if character operations are always on the global DB.
        if not self.unified_db:
            logger.error("Unified database not available in add_character_card.")
            raise CharactersRAGDBError("Unified database not available.")
        logger.debug(f"Service: Adding character for user '{user_id}' with data: {character_data.get('name')}")
        # ChaChaNotes_DB.add_character_card expects user_id as a named argument.
        return self.unified_db.add_character_card(character_data=character_data, user_id=user_id)

    def update_character_card(self, character_id: str, user_id: str, update_data: Dict[str, Any], expected_version: Optional[int]) -> Optional[Dict[str, Any]]:
        """Updates an existing character card for the specified user with optimistic locking."""
        if not self.unified_db:
            logger.error("Unified database not available in update_character_card.")
            raise CharactersRAGDBError("Unified database not available.")
        logger.debug(f"Service: Updating character ID '{character_id}' for user '{user_id}'. Version: {expected_version}")
        # ChaChaNotes_DB.update_character_card expects user_id.
        return self.unified_db.update_character_card(
            character_id=character_id,
            user_id=user_id,
            update_data=update_data,
            expected_version=expected_version
        )

    # --- Resource Management ---

    def close_all_user_connections(self):
        with self._db_lock:
            logger.info(f"Closing all {len(self._db_instances)} cached user-context DB instances.")
            for user_id, db_instance in self._db_instances.items():
                try:
                    # Each db_instance is a CharactersRAGDB, call its close_connection
                    db_instance.close_connection()
                    logger.debug(f"Closed DB instance for user context '{user_id}'.")
                except Exception as e:
                    logger.error(f"Error closing DB instance for user context '{user_id}': {e}", exc_info=True)
            self._db_instances.clear()
        logger.info("All cached user-context DB instances have been processed for closure.")

    def close_user_connection(self, user_id: str):
        with self._db_lock:
            if user_id in self._db_instances:
                db_instance = self._db_instances.pop(user_id)
                try:
                    db_instance.close_connection()
                    logger.info(f"Closed and removed DB instance for user context '{user_id}'.")
                except Exception as e:
                    logger.error(f"Error closing DB instance for user context '{user_id}': {e}", exc_info=True)
            else:
                logger.debug(f"No active DB instance found in cache for user context '{user_id}' to close.")

    # --- Sync-Related Methods ---
    
    def get_notes_for_sync(self, user_id: str, sync_root_folder: Path = None) -> List[Dict[str, Any]]:
        """
        Get notes that are configured for external sync.
        
        Args:
            user_id: User ID for database context
            sync_root_folder: Optional filter by sync root folder
            
        Returns:
            List of note dictionaries with sync metadata
        """
        db = self._get_db(user_id)
        
        try:
            with db.transaction() as conn:
                if sync_root_folder:
                    cursor = conn.execute("""
                        SELECT id, title, content, version, file_path_on_disk,
                               relative_file_path_on_disk, sync_root_folder,
                               last_synced_disk_file_hash, last_synced_disk_file_mtime,
                               is_externally_synced, sync_strategy, sync_excluded,
                               file_extension, last_modified, created_at
                        FROM notes
                        WHERE deleted = 0 AND sync_root_folder = ?
                        ORDER BY last_modified DESC
                    """, (str(sync_root_folder),))
                else:
                    cursor = conn.execute("""
                        SELECT id, title, content, version, file_path_on_disk,
                               relative_file_path_on_disk, sync_root_folder,
                               last_synced_disk_file_hash, last_synced_disk_file_mtime,
                               is_externally_synced, sync_strategy, sync_excluded,
                               file_extension, last_modified, created_at
                        FROM notes
                        WHERE deleted = 0 AND is_externally_synced = 1
                        ORDER BY last_modified DESC
                    """)
                
                notes = []
                for row in cursor:
                    notes.append({
                        'id': row[0],
                        'title': row[1],
                        'content': row[2],
                        'version': row[3],
                        'file_path_on_disk': row[4],
                        'relative_file_path_on_disk': row[5],
                        'sync_root_folder': row[6],
                        'last_synced_disk_file_hash': row[7],
                        'last_synced_disk_file_mtime': row[8],
                        'is_externally_synced': row[9],
                        'sync_strategy': row[10],
                        'sync_excluded': row[11],
                        'file_extension': row[12],
                        'last_modified': row[13],
                        'created_at': row[14]
                    })
                
                return notes
                
        except Exception as e:
            logger.error(f"Error getting notes for sync: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Failed to get notes for sync: {e}") from e
    
    def get_unsynced_notes(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get notes that are not currently synced to any external file.
        
        Args:
            user_id: User ID for database context
            limit: Maximum number of notes to return
            
        Returns:
            List of note dictionaries
        """
        db = self._get_db(user_id)
        
        try:
            with db.transaction() as conn:
                cursor = conn.execute("""
                    SELECT id, title, content, version, last_modified, created_at
                    FROM notes
                    WHERE deleted = 0 
                      AND (is_externally_synced = 0 OR is_externally_synced IS NULL)
                      AND (sync_excluded = 0 OR sync_excluded IS NULL)
                    ORDER BY last_modified DESC
                    LIMIT ?
                """, (limit,))
                
                notes = []
                for row in cursor:
                    notes.append({
                        'id': row[0],
                        'title': row[1],
                        'content': row[2],
                        'version': row[3],
                        'last_modified': row[4],
                        'created_at': row[5]
                    })
                
                return notes
                
        except Exception as e:
            logger.error(f"Error getting unsynced notes: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Failed to get unsynced notes: {e}") from e
    
    def update_note_sync_metadata(self, user_id: str, note_id: str, sync_metadata: Dict[str, Any],
                                expected_version: int) -> bool:
        """
        Update sync-related metadata for a note.
        
        Args:
            user_id: User ID for database context
            note_id: Note ID to update
            sync_metadata: Dictionary containing sync fields to update
            expected_version: Expected version for optimistic locking
            
        Returns:
            True if updated successfully, False otherwise
        """
        db = self._get_db(user_id)
        
        try:
            with db.transaction() as conn:
                # Build update fields from sync_metadata
                update_fields = []
                values = []
                
                allowed_fields = {
                    'file_path_on_disk', 'relative_file_path_on_disk', 'sync_root_folder',
                    'last_synced_disk_file_hash', 'last_synced_disk_file_mtime',
                    'is_externally_synced', 'sync_strategy', 'sync_excluded', 'file_extension'
                }
                
                for field, value in sync_metadata.items():
                    if field in allowed_fields:
                        update_fields.append(f"{field} = ?")
                        values.append(value)
                
                if not update_fields:
                    logger.warning("No valid sync metadata fields to update")
                    return False
                
                # Add version increment and timestamp
                update_fields.append("version = ?")
                values.append(expected_version + 1)
                
                update_fields.append("last_modified = CURRENT_TIMESTAMP")
                
                # Add WHERE clause parameters
                values.extend([note_id, expected_version])
                
                query = f"""
                    UPDATE notes
                    SET {', '.join(update_fields)}
                    WHERE id = ? AND version = ? AND deleted = 0
                """
                
                cursor = conn.execute(query, values)
                
                if cursor.rowcount == 0:
                    logger.warning(f"No rows updated for note {note_id} - version mismatch or deleted")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating note sync metadata: {e}", exc_info=True)
            return False
    
    def link_note_to_file(self, user_id: str, note_id: str, file_path: Path,
                         sync_root_folder: Path, sync_strategy: str = 'bidirectional') -> bool:
        """
        Link a note to an external file for syncing.
        
        Args:
            user_id: User ID for database context
            note_id: Note ID to link
            file_path: Absolute path to the file
            sync_root_folder: Root folder for sync
            sync_strategy: Sync strategy ('disk_to_db', 'db_to_disk', 'bidirectional')
            
        Returns:
            True if linked successfully
        """
        # Get current note version
        note = self.get_note_by_id(user_id, note_id)
        if not note:
            logger.error(f"Note {note_id} not found")
            return False
        
        try:
            relative_path = file_path.relative_to(sync_root_folder)
        except ValueError:
            logger.error(f"File {file_path} is not under sync root {sync_root_folder}")
            return False
        
        sync_metadata = {
            'file_path_on_disk': str(file_path),
            'relative_file_path_on_disk': str(relative_path),
            'sync_root_folder': str(sync_root_folder),
            'is_externally_synced': True,
            'sync_strategy': sync_strategy,
            'file_extension': file_path.suffix
        }
        
        return self.update_note_sync_metadata(user_id, note_id, sync_metadata, note['version'])
    
    def unlink_note_from_file(self, user_id: str, note_id: str) -> bool:
        """
        Unlink a note from its external file.
        
        Args:
            user_id: User ID for database context
            note_id: Note ID to unlink
            
        Returns:
            True if unlinked successfully
        """
        # Get current note version
        note = self.get_note_by_id(user_id, note_id)
        if not note:
            logger.error(f"Note {note_id} not found")
            return False
        
        sync_metadata = {
            'file_path_on_disk': None,
            'relative_file_path_on_disk': None,
            'last_synced_disk_file_hash': None,
            'last_synced_disk_file_mtime': None,
            'is_externally_synced': False
        }
        
        return self.update_note_sync_metadata(user_id, note_id, sync_metadata, note['version'])
    
    def get_sync_status(self, user_id: str, note_id: str) -> Dict[str, Any]:
        """
        Get the sync status for a specific note.
        
        Args:
            user_id: User ID for database context
            note_id: Note ID to check
            
        Returns:
            Dictionary containing sync status information
        """
        db = self._get_db(user_id)
        
        try:
            with db.transaction() as conn:
                cursor = conn.execute("""
                    SELECT id, title, version, file_path_on_disk,
                           relative_file_path_on_disk, sync_root_folder,
                           last_synced_disk_file_hash, last_synced_disk_file_mtime,
                           is_externally_synced, sync_strategy, sync_excluded,
                           file_extension, last_modified, created_at
                    FROM notes
                    WHERE id = ? AND deleted = 0
                """, (note_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                return {
                    'id': row[0],
                    'title': row[1],
                    'version': row[2],
                    'file_path_on_disk': row[3],
                    'relative_file_path_on_disk': row[4],
                    'sync_root_folder': row[5],
                    'last_synced_disk_file_hash': row[6],
                    'last_synced_disk_file_mtime': row[7],
                    'is_externally_synced': row[8],
                    'sync_strategy': row[9],
                    'sync_excluded': row[10],
                    'file_extension': row[11],
                    'last_modified': row[12],
                    'created_at': row[13]
                }
                
        except Exception as e:
            logger.error(f"Error getting sync status for note {note_id}: {e}", exc_info=True)
            raise CharactersRAGDBError(f"Failed to get sync status: {e}") from e
    
    def set_note_sync_enabled(self, user_id: str, note_id: str, enabled: bool,
                             sync_strategy: str = 'bidirectional') -> bool:
        """
        Enable or disable sync for a specific note.
        
        Args:
            user_id: User ID for database context
            note_id: Note ID to update
            enabled: Whether to enable or disable sync
            sync_strategy: Sync strategy when enabling
            
        Returns:
            True if updated successfully
        """
        # Get current note version
        note = self.get_note_by_id(user_id, note_id)
        if not note:
            logger.error(f"Note {note_id} not found")
            return False
        
        sync_metadata = {
            'is_externally_synced': enabled
        }
        
        if enabled:
            sync_metadata['sync_strategy'] = sync_strategy
        
        return self.update_note_sync_metadata(user_id, note_id, sync_metadata, note['version'])

#
# End of Notes_Library.py
#######################################################################################################################
