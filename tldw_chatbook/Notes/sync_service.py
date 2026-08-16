# sync_service.py
# Description: Service layer for note synchronization operations
#
# Imports
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime

#
# Third-Party Imports
from loguru import logger

#
# Local Imports
from .sync_engine import (
    NotesSyncEngine,
    SyncDirection,
    ConflictResolution,
    SyncProgress,
)
from .Notes_Library import NotesInteropService
from ..DB.ChaChaNotes_DB import CharactersRAGDB
#
########################################################################################################################
#
# Classes:


class NotesSyncService:
    """High-level service for note synchronization."""

    def __init__(
        self,
        notes_service: NotesInteropService,
        db: CharactersRAGDB,
    ):
        """
        Initialize sync service.

        Args:
            notes_service: Notes service for database operations
            db: Database instance
        """
        self.notes_service = notes_service
        self.db = db
        self.sync_engine = NotesSyncEngine(notes_service, db)

    async def sync_folder(
        self,
        root_folder: Path,
        user_id: str,
        direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        conflict_resolution: ConflictResolution = ConflictResolution.ASK,
        extensions: List[str] = None,
        progress_callback: Optional[Callable[[SyncProgress], None]] = None,
    ) -> Tuple[str, SyncProgress]:
        """Execute one-time sync for a folder."""
        self.sync_engine.progress_callback = progress_callback

        return await self.sync_engine.sync(
            root_path=root_folder,
            user_id=user_id,
            direction=direction,
            conflict_resolution=conflict_resolution,
            extensions=extensions,
        )

    def get_sync_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent sync session history."""
        history = []

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT session_id, sync_root_folder, sync_direction, 
                       conflict_resolution, started_at, completed_at,
                       status, total_files, processed_files, conflicts_found,
                       errors_count, summary
                FROM sync_sessions
                ORDER BY started_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            for row in cursor:
                session_data = {
                    "session_id": row[0],
                    "sync_root_folder": row[1],
                    "sync_direction": row[2],
                    "conflict_resolution": row[3],
                    "started_at": row[4],
                    "completed_at": row[5],
                    "status": row[6],
                    "total_files": row[7],
                    "processed_files": row[8],
                    "conflicts_found": row[9],
                    "errors_count": row[10],
                    "summary": json.loads(row[11]) if row[11] else None,
                }
                history.append(session_data)

        return history

    def get_conflicts_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conflicts for a specific sync session."""
        conflicts = []

        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT id, note_id, file_path, conflict_type,
                       db_content_hash, disk_content_hash,
                       db_modified_time, disk_modified_time,
                       resolution, resolved_at, created_at
                FROM sync_conflicts
                WHERE session_id = ?
                ORDER BY created_at DESC
            """,
                (session_id,),
            )

            for row in cursor:
                conflict_data = {
                    "id": row[0],
                    "note_id": row[1],
                    "file_path": row[2],
                    "conflict_type": row[3],
                    "db_content_hash": row[4],
                    "disk_content_hash": row[5],
                    "db_modified_time": row[6],
                    "disk_modified_time": row[7],
                    "resolution": row[8],
                    "resolved_at": row[9],
                    "created_at": row[10],
                }
                conflicts.append(conflict_data)

        return conflicts

    def resolve_conflict(self, conflict_id: int, resolution: str, user_id: str) -> bool:
        """
        Resolve a sync conflict.

        Args:
            conflict_id: ID of the conflict
            resolution: One of 'use_db', 'use_disk', 'merge', 'skip'
            user_id: User ID for database operations

        Returns:
            True if resolved successfully
        """
        try:
            with self.db.transaction() as conn:
                # Get conflict details
                cursor = conn.execute(
                    """
                    SELECT note_id, file_path, conflict_type, session_id
                    FROM sync_conflicts
                    WHERE id = ? AND resolution IS NULL
                """,
                    (conflict_id,),
                )

                row = cursor.fetchone()
                if not row:
                    logger.warning(
                        f"Conflict {conflict_id} not found or already resolved"
                    )
                    return False

                note_id, file_path, conflict_type, session_id = row

                # Update conflict resolution
                conn.execute(
                    """
                    UPDATE sync_conflicts
                    SET resolution = ?, resolved_at = ?
                    WHERE id = ?
                """,
                    (resolution, datetime.now().isoformat(), conflict_id),
                )

                # TODO: Implement actual resolution logic based on resolution type
                # This would involve updating the note or file based on the resolution

                return True

        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}")
            return False

    def get_notes_sync_status(
        self, root_folder: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Get sync status for notes, optionally filtered by root folder."""
        notes_status = []

        query = """
            SELECT n.id, n.title, n.file_path_on_disk, n.sync_root_folder,
                   n.last_synced_disk_file_hash, n.last_synced_disk_file_mtime,
                   n.is_externally_synced, n.sync_strategy, n.sync_excluded,
                   n.last_modified, n.content
            FROM notes n
            WHERE n.deleted = 0 AND n.is_externally_synced = 1
        """

        params = []
        if root_folder:
            query += " AND n.sync_root_folder = ?"
            params.append(str(root_folder))

        with self.db.transaction() as conn:
            cursor = conn.execute(query, params)

            for row in cursor:
                note_data = {
                    "id": row[0],
                    "title": row[1],
                    "file_path": row[2],
                    "sync_root_folder": row[3],
                    "last_synced_hash": row[4],
                    "last_synced_mtime": row[5],
                    "is_synced": row[6],
                    "sync_strategy": row[7],
                    "sync_excluded": row[8],
                    "last_modified": row[9],
                    "current_hash": self.sync_engine._calculate_hash(row[10]),
                }

                # Check sync status
                if note_data["file_path"] and Path(note_data["file_path"]).exists():
                    try:
                        file_content = Path(note_data["file_path"]).read_text(
                            encoding="utf-8"
                        )
                        file_hash = self.sync_engine._calculate_hash(file_content)

                        if file_hash != note_data["last_synced_hash"]:
                            if (
                                note_data["current_hash"]
                                != note_data["last_synced_hash"]
                            ):
                                note_data["sync_status"] = "conflict"
                            else:
                                note_data["sync_status"] = "file_changed"
                        elif note_data["current_hash"] != note_data["last_synced_hash"]:
                            note_data["sync_status"] = "db_changed"
                        else:
                            note_data["sync_status"] = "synced"
                    except (OSError, IOError, UnicodeDecodeError) as e:
                        logger.warning(
                            f"Error reading file {note_data['file_path']}: {e}"
                        )
                        note_data["sync_status"] = "file_error"
                else:
                    note_data["sync_status"] = "file_missing"

                notes_status.append(note_data)

        return notes_status


#
# End of sync_service.py
########################################################################################################################
