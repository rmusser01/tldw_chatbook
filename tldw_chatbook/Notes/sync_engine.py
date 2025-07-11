# sync_engine.py
# Description: Engine for bi-directional file synchronization of notes
#
# Imports
import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from ..DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError
from .Notes_Library import NotesInteropService
from ..Metrics.metrics_logger import log_counter, log_histogram
#
########################################################################################################################
#
# Classes and Functions:

class SyncDirection(Enum):
    """Enumeration for sync directions."""
    DISK_TO_DB = "disk_to_db"
    DB_TO_DISK = "db_to_disk"
    BIDIRECTIONAL = "bidirectional"


class ConflictResolution(Enum):
    """Enumeration for conflict resolution strategies."""
    ASK = "ask"
    DISK_WINS = "disk_wins"
    DB_WINS = "db_wins"
    NEWER_WINS = "newer_wins"


class SyncStatus(Enum):
    """Enumeration for sync session status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SyncFileInfo:
    """Information about a file for syncing."""
    absolute_path: Path
    relative_path: Path
    content: str
    content_hash: str
    mtime: float
    extension: str


@dataclass
class SyncConflict:
    """Represents a sync conflict."""
    note_id: Optional[str]
    file_path: Path
    conflict_type: str
    db_content: Optional[str] = None
    disk_content: Optional[str] = None
    db_hash: Optional[str] = None
    disk_hash: Optional[str] = None
    db_modified: Optional[datetime] = None
    disk_modified: Optional[float] = None


@dataclass
class SyncProgress:
    """Tracks sync operation progress."""
    total_files: int = 0
    processed_files: int = 0
    conflicts: List[SyncConflict] = field(default_factory=list)
    errors: List[Tuple[str, Exception]] = field(default_factory=list)
    created_notes: List[str] = field(default_factory=list)
    updated_notes: List[str] = field(default_factory=list)
    created_files: List[Path] = field(default_factory=list)
    updated_files: List[Path] = field(default_factory=list)
    skipped_items: List[Tuple[str, str]] = field(default_factory=list)  # (item, reason)


class NotesSyncEngine:
    """Engine for synchronizing notes between database and file system."""
    
    def __init__(self, 
                 notes_service: NotesInteropService,
                 db: CharactersRAGDB,
                 progress_callback: Optional[Callable[[SyncProgress], None]] = None):
        """
        Initialize the sync engine.
        
        Args:
            notes_service: The notes service for database operations
            db: Direct database access for sync-specific operations
            progress_callback: Optional callback for progress updates
        """
        self.notes_service = notes_service
        self.db = db
        self.progress_callback = progress_callback
        self._active_sessions: Dict[str, SyncProgress] = {}
        self._cancelled_sessions: Set[str] = set()
        
    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_file_info(self, file_path: Path, root_path: Path) -> Optional[SyncFileInfo]:
        """Get file information for syncing."""
        try:
            content = file_path.read_text(encoding='utf-8')
            relative_path = file_path.relative_to(root_path)
            
            return SyncFileInfo(
                absolute_path=file_path,
                relative_path=relative_path,
                content=content,
                content_hash=self._calculate_hash(content),
                mtime=file_path.stat().st_mtime,
                extension=file_path.suffix.lower()
            )
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return None
    
    def _scan_directory(self, root_path: Path, extensions: List[str] = None) -> Dict[Path, SyncFileInfo]:
        """
        Scan directory for files to sync.
        
        Args:
            root_path: Root directory to scan
            extensions: List of file extensions to include (e.g., ['.md', '.txt'])
                       If None, defaults to ['.md', '.txt']
        
        Returns:
            Dictionary mapping relative paths to file info
        """
        start_time = time.time()
        log_counter("sync_engine_scan_directory_attempt", labels={"extensions_count": str(len(extensions) if extensions else 2)})
        
        if extensions is None:
            extensions = ['.md', '.txt']
        
        files_map = {}
        files_scanned = 0
        files_failed = 0
        
        for ext in extensions:
            for file_path in root_path.rglob(f'*{ext}'):
                if file_path.is_file():
                    files_scanned += 1
                    file_info = self._get_file_info(file_path, root_path)
                    if file_info:
                        files_map[file_info.relative_path] = file_info
                    else:
                        files_failed += 1
        
        # Log metrics
        duration = time.time() - start_time
        log_histogram("sync_engine_scan_directory_duration", duration)
        log_histogram("sync_engine_scan_files_found", len(files_map))
        log_histogram("sync_engine_scan_files_failed", files_failed)
        log_counter("sync_engine_scan_directory_success", labels={
            "files_found": str(len(files_map)),
            "files_failed": str(files_failed)
        })
        
        logger.info(f"Scanned {len(files_map)} files in {root_path}")
        return files_map
    
    def _get_synced_notes_for_root(self, root_path: Path, user_id: str) -> Dict[Path, Dict[str, Any]]:
        """Get all notes that are synced to the given root folder."""
        db_notes_map = {}
        
        with self.db.transaction() as conn:
            cursor = conn.execute("""
                SELECT id, title, content, version, relative_file_path_on_disk,
                       last_synced_disk_file_hash, last_synced_disk_file_mtime,
                       last_modified, file_extension, sync_strategy, sync_excluded
                FROM notes
                WHERE deleted = 0 
                  AND sync_root_folder = ? 
                  AND is_externally_synced = 1
                  AND sync_excluded = 0
            """, (str(root_path),))
            
            for row in cursor:
                note_data = {
                    'id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'version': row[3],
                    'relative_file_path_on_disk': row[4],
                    'last_synced_disk_file_hash': row[5],
                    'last_synced_disk_file_mtime': row[6],
                    'last_modified': row[7],
                    'file_extension': row[8] or '.md',
                    'sync_strategy': row[9],
                    'sync_excluded': row[10]
                }
                
                if note_data['relative_file_path_on_disk']:
                    rel_path = Path(note_data['relative_file_path_on_disk'])
                    note_data['content_hash'] = self._calculate_hash(note_data['content'])
                    db_notes_map[rel_path] = note_data
        
        logger.info(f"Found {len(db_notes_map)} synced notes for root {root_path}")
        return db_notes_map
    
    def _create_sync_session(self, sync_root: Path, direction: SyncDirection, 
                           conflict_resolution: ConflictResolution, user_id: str) -> str:
        """Create a new sync session in the database."""
        session_id = str(uuid.uuid4())
        
        with self.db.transaction() as conn:
            conn.execute("""
                INSERT INTO sync_sessions 
                (session_id, sync_root_folder, sync_direction, conflict_resolution, 
                 status, client_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, str(sync_root), direction.value, conflict_resolution.value,
                  SyncStatus.RUNNING.value, user_id))
        
        self._active_sessions[session_id] = SyncProgress()
        return session_id
    
    def _update_sync_session(self, session_id: str, progress: SyncProgress, 
                           status: SyncStatus, summary: Optional[Dict] = None):
        """Update sync session in the database."""
        with self.db.transaction() as conn:
            update_data = {
                'processed_files': progress.processed_files,
                'conflicts_found': len(progress.conflicts),
                'errors_count': len(progress.errors),
                'status': status.value
            }
            
            if status in (SyncStatus.COMPLETED, SyncStatus.FAILED, SyncStatus.CANCELLED):
                update_data['completed_at'] = datetime.now(timezone.utc).isoformat()
            
            if summary:
                update_data['summary'] = json.dumps(summary)
            
            # Build UPDATE query
            set_clauses = [f"{k} = ?" for k in update_data.keys()]
            values = list(update_data.values()) + [session_id]
            
            conn.execute(f"""
                UPDATE sync_sessions 
                SET {', '.join(set_clauses)}
                WHERE session_id = ?
            """, values)
    
    def _record_conflict(self, session_id: str, conflict: SyncConflict):
        """Record a sync conflict in the database."""
        with self.db.transaction() as conn:
            conn.execute("""
                INSERT INTO sync_conflicts
                (session_id, note_id, file_path, conflict_type,
                 db_content_hash, disk_content_hash, 
                 db_modified_time, disk_modified_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, conflict.note_id, str(conflict.file_path),
                  conflict.conflict_type, conflict.db_hash, conflict.disk_hash,
                  conflict.db_modified.isoformat() if conflict.db_modified else None,
                  conflict.disk_modified))
    
    def _update_note_sync_metadata(self, note_id: str, file_info: SyncFileInfo,
                                 root_path: Path, user_id: str, version: int):
        """Update note's sync metadata after successful sync."""
        with self.db.transaction() as conn:
            now = datetime.now(timezone.utc).isoformat()
            new_version = version + 1
            
            cursor = conn.execute("""
                UPDATE notes
                SET file_path_on_disk = ?,
                    relative_file_path_on_disk = ?,
                    sync_root_folder = ?,
                    last_synced_disk_file_hash = ?,
                    last_synced_disk_file_mtime = ?,
                    is_externally_synced = 1,
                    file_extension = ?,
                    last_modified = ?,
                    version = ?
                WHERE id = ? AND version = ?
            """, (str(file_info.absolute_path), str(file_info.relative_path),
                  str(root_path), file_info.content_hash, file_info.mtime,
                  file_info.extension, now, new_version, note_id, version))
            
            if cursor.rowcount == 0:
                raise ConflictError(f"Version mismatch updating sync metadata for note {note_id}")
    
    def _unlink_note_from_sync(self, note_id: str, version: int):
        """Remove sync metadata from a note."""
        with self.db.transaction() as conn:
            now = datetime.now(timezone.utc).isoformat()
            new_version = version + 1
            
            cursor = conn.execute("""
                UPDATE notes
                SET file_path_on_disk = NULL,
                    relative_file_path_on_disk = NULL,
                    last_synced_disk_file_hash = NULL,
                    last_synced_disk_file_mtime = NULL,
                    is_externally_synced = 0,
                    last_modified = ?,
                    version = ?
                WHERE id = ? AND version = ?
            """, (now, new_version, note_id, version))
            
            if cursor.rowcount == 0:
                raise ConflictError(f"Version mismatch unlinking note {note_id}")
    
    def cancel_sync(self, session_id: str):
        """Cancel an active sync session."""
        self._cancelled_sessions.add(session_id)
        logger.info(f"Sync session {session_id} marked for cancellation")
    
    def is_cancelled(self, session_id: str) -> bool:
        """Check if a sync session has been cancelled."""
        return session_id in self._cancelled_sessions
    
    async def sync(self, 
                   root_path: Path,
                   user_id: str,
                   direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
                   conflict_resolution: ConflictResolution = ConflictResolution.ASK,
                   extensions: Optional[List[str]] = None,
                   post_sync_cleanup: bool = False) -> Tuple[str, SyncProgress]:
        """
        Main sync method.
        
        Args:
            root_path: Root directory for sync
            user_id: User ID for database operations
            direction: Sync direction
            conflict_resolution: How to handle conflicts
            extensions: File extensions to sync
            post_sync_cleanup: Whether to unlink files after sync
            
        Returns:
            Tuple of (session_id, progress)
        """
        start_time = time.time()
        log_counter("sync_engine_sync_attempt", labels={
            "direction": direction.value,
            "conflict_resolution": conflict_resolution.value,
            "post_sync_cleanup": str(post_sync_cleanup)
        })
        
        session_id = self._create_sync_session(root_path, direction, conflict_resolution, user_id)
        progress = self._active_sessions[session_id]
        
        try:
            logger.info(f"Starting sync session {session_id}: {root_path}, {direction.value}")
            
            # Scan directory and get DB notes
            disk_files = self._scan_directory(root_path, extensions)
            db_notes = self._get_synced_notes_for_root(root_path, user_id)
            
            progress.total_files = len(set(disk_files.keys()) | set(db_notes.keys()))
            
            if direction == SyncDirection.DISK_TO_DB:
                await self._sync_disk_to_db(session_id, root_path, disk_files, db_notes, 
                                          conflict_resolution, user_id, progress)
            elif direction == SyncDirection.DB_TO_DISK:
                await self._sync_db_to_disk(session_id, root_path, disk_files, db_notes,
                                          conflict_resolution, user_id, progress)
            else:  # BIDIRECTIONAL
                await self._sync_bidirectional(session_id, root_path, disk_files, db_notes,
                                             conflict_resolution, user_id, progress)
            
            # Update session status
            status = SyncStatus.CANCELLED if self.is_cancelled(session_id) else SyncStatus.COMPLETED
            summary = {
                'created_notes': len(progress.created_notes),
                'updated_notes': len(progress.updated_notes),
                'created_files': len(progress.created_files),
                'updated_files': len(progress.updated_files),
                'conflicts': len(progress.conflicts),
                'errors': len(progress.errors),
                'skipped': len(progress.skipped_items)
            }
            
            self._update_sync_session(session_id, progress, status, summary)
            
            # Cleanup
            if session_id in self._cancelled_sessions:
                self._cancelled_sessions.remove(session_id)
            
            # Log success metrics
            duration = time.time() - start_time
            log_histogram("sync_engine_sync_duration", duration, labels={
                "status": "success",
                "direction": direction.value
            })
            log_counter("sync_engine_sync_complete", labels={
                "direction": direction.value,
                "created_notes": str(summary['created_notes']),
                "updated_notes": str(summary['updated_notes']),
                "created_files": str(summary['created_files']),
                "updated_files": str(summary['updated_files']),
                "conflicts": str(summary['conflicts']),
                "errors": str(summary['errors'])
            })
            log_histogram("sync_engine_sync_conflicts", len(progress.conflicts))
            log_histogram("sync_engine_sync_errors", len(progress.errors))
            
            logger.info(f"Sync session {session_id} completed: {summary}")
            
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("sync_engine_sync_duration", duration, labels={
                "status": "error",
                "direction": direction.value
            })
            log_counter("sync_engine_sync_error", labels={
                "direction": direction.value,
                "error_type": type(e).__name__
            })
            
            logger.error(f"Sync session {session_id} failed: {e}", exc_info=True)
            self._update_sync_session(session_id, progress, SyncStatus.FAILED)
            raise
        finally:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
        
        return session_id, progress
    
    async def _sync_disk_to_db(self, session_id: str, root_path: Path,
                              disk_files: Dict[Path, SyncFileInfo],
                              db_notes: Dict[Path, Dict[str, Any]],
                              conflict_resolution: ConflictResolution,
                              user_id: str,
                              progress: SyncProgress):
        """Sync from disk to database."""
        for rel_path, file_info in disk_files.items():
            if self.is_cancelled(session_id):
                break
            
            db_note = db_notes.get(rel_path)
            
            try:
                if not db_note:
                    # New file on disk -> Create note in DB
                    title = file_info.absolute_path.stem
                    note_id = self.notes_service.add_note(
                        user_id=user_id,
                        title=title,
                        content=file_info.content
                    )
                    
                    if note_id:
                        self._update_note_sync_metadata(note_id, file_info, root_path, user_id, 1)
                        progress.created_notes.append(note_id)
                        logger.info(f"Created note from file: {rel_path}")
                    
                elif file_info.content_hash != db_note.get('last_synced_disk_file_hash'):
                    # File changed on disk -> Update note in DB
                    success = self.notes_service.update_note(
                        user_id=user_id,
                        note_id=db_note['id'],
                        update_data={
                            'content': file_info.content,
                            'title': file_info.absolute_path.stem
                        },
                        expected_version=db_note['version']
                    )
                    
                    if success:
                        # Get updated version
                        updated_note = self.notes_service.get_note_by_id(user_id, db_note['id'])
                        if updated_note:
                            self._update_note_sync_metadata(
                                db_note['id'], file_info, root_path, user_id, 
                                updated_note['version']
                            )
                            progress.updated_notes.append(db_note['id'])
                            logger.info(f"Updated note from file: {rel_path}")
                
            except Exception as e:
                logger.error(f"Error syncing {rel_path}: {e}")
                progress.errors.append((str(rel_path), e))
            
            progress.processed_files += 1
            if self.progress_callback:
                self.progress_callback(progress)
        
        # Check for notes that no longer have files
        for rel_path, db_note in db_notes.items():
            if rel_path not in disk_files:
                conflict = SyncConflict(
                    note_id=db_note['id'],
                    file_path=rel_path,
                    conflict_type='deleted_on_disk',
                    db_content=db_note['content'],
                    db_hash=db_note.get('content_hash')
                )
                progress.conflicts.append(conflict)
                self._record_conflict(session_id, conflict)
                
                # Auto-unlink if not asking
                if conflict_resolution != ConflictResolution.ASK:
                    try:
                        self._unlink_note_from_sync(db_note['id'], db_note['version'])
                        logger.info(f"Unlinked note {db_note['id']} - file deleted on disk")
                    except Exception as e:
                        logger.error(f"Error unlinking note {db_note['id']}: {e}")
    
    async def _sync_db_to_disk(self, session_id: str, root_path: Path,
                              disk_files: Dict[Path, SyncFileInfo],
                              db_notes: Dict[Path, Dict[str, Any]],
                              conflict_resolution: ConflictResolution,
                              user_id: str,
                              progress: SyncProgress):
        """Sync from database to disk."""
        for rel_path, db_note in db_notes.items():
            if self.is_cancelled(session_id):
                break
            
            file_path = root_path / rel_path
            db_content_hash = db_note['content_hash']
            
            try:
                if rel_path not in disk_files:
                    # Note in DB but no file -> Create file
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(db_note['content'], encoding='utf-8')
                    
                    # Update sync metadata
                    new_file_info = self._get_file_info(file_path, root_path)
                    if new_file_info:
                        self._update_note_sync_metadata(
                            db_note['id'], new_file_info, root_path, user_id,
                            db_note['version']
                        )
                        progress.created_files.append(file_path)
                        logger.info(f"Created file from note: {rel_path}")
                        
                elif db_content_hash != db_note.get('last_synced_disk_file_hash'):
                    # Note changed in DB -> Update file
                    disk_file = disk_files[rel_path]
                    
                    if db_content_hash != disk_file.content_hash:
                        # Check for conflict
                        if disk_file.content_hash != db_note.get('last_synced_disk_file_hash'):
                            # Both changed - conflict!
                            conflict = SyncConflict(
                                note_id=db_note['id'],
                                file_path=rel_path,
                                conflict_type='both_changed',
                                db_content=db_note['content'],
                                disk_content=disk_file.content,
                                db_hash=db_content_hash,
                                disk_hash=disk_file.content_hash
                            )
                            progress.conflicts.append(conflict)
                            self._record_conflict(session_id, conflict)
                            
                            # Resolve based on strategy
                            if conflict_resolution == ConflictResolution.DB_WINS:
                                file_path.write_text(db_note['content'], encoding='utf-8')
                                progress.updated_files.append(file_path)
                        else:
                            # Only DB changed
                            file_path.write_text(db_note['content'], encoding='utf-8')
                            progress.updated_files.append(file_path)
                            
                            # Update sync metadata
                            new_file_info = self._get_file_info(file_path, root_path)
                            if new_file_info:
                                self._update_note_sync_metadata(
                                    db_note['id'], new_file_info, root_path, user_id,
                                    db_note['version']
                                )
                            logger.info(f"Updated file from note: {rel_path}")
                            
            except Exception as e:
                logger.error(f"Error syncing note {db_note['id']} to disk: {e}")
                progress.errors.append((f"Note {db_note['id']}", e))
            
            progress.processed_files += 1
            if self.progress_callback:
                self.progress_callback(progress)
    
    async def _sync_bidirectional(self, session_id: str, root_path: Path,
                                 disk_files: Dict[Path, SyncFileInfo],
                                 db_notes: Dict[Path, Dict[str, Any]],
                                 conflict_resolution: ConflictResolution,
                                 user_id: str,
                                 progress: SyncProgress):
        """Bidirectional sync between disk and database."""
        all_paths = set(disk_files.keys()) | set(db_notes.keys())
        
        for rel_path in all_paths:
            if self.is_cancelled(session_id):
                break
            
            disk_file = disk_files.get(rel_path)
            db_note = db_notes.get(rel_path)
            
            try:
                if disk_file and not db_note:
                    # Only on disk -> Create in DB
                    title = disk_file.absolute_path.stem
                    note_id = self.notes_service.add_note(
                        user_id=user_id,
                        title=title,
                        content=disk_file.content
                    )
                    
                    if note_id:
                        self._update_note_sync_metadata(note_id, disk_file, root_path, user_id, 1)
                        progress.created_notes.append(note_id)
                        logger.info(f"Created note from file: {rel_path}")
                        
                elif not disk_file and db_note:
                    # Only in DB -> Create on disk or handle deletion
                    conflict = SyncConflict(
                        note_id=db_note['id'],
                        file_path=rel_path,
                        conflict_type='deleted_on_disk',
                        db_content=db_note['content']
                    )
                    progress.conflicts.append(conflict)
                    self._record_conflict(session_id, conflict)
                    
                    # Auto-resolve based on strategy
                    if conflict_resolution == ConflictResolution.DB_WINS:
                        # Recreate file
                        file_path = root_path / rel_path
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_path.write_text(db_note['content'], encoding='utf-8')
                        progress.created_files.append(file_path)
                        
                elif disk_file and db_note:
                    # Exists in both - check for changes
                    db_content_hash = db_note['content_hash']
                    disk_hash = disk_file.content_hash
                    last_synced_hash = db_note.get('last_synced_disk_file_hash')
                    
                    db_changed = db_content_hash != last_synced_hash
                    disk_changed = disk_hash != last_synced_hash
                    
                    if db_changed and not disk_changed:
                        # Only DB changed -> Update disk
                        disk_file.absolute_path.write_text(db_note['content'], encoding='utf-8')
                        new_file_info = self._get_file_info(disk_file.absolute_path, root_path)
                        if new_file_info:
                            self._update_note_sync_metadata(
                                db_note['id'], new_file_info, root_path, user_id,
                                db_note['version']
                            )
                        progress.updated_files.append(disk_file.absolute_path)
                        
                    elif not db_changed and disk_changed:
                        # Only disk changed -> Update DB
                        success = self.notes_service.update_note(
                            user_id=user_id,
                            note_id=db_note['id'],
                            update_data={
                                'content': disk_file.content,
                                'title': disk_file.absolute_path.stem
                            },
                            expected_version=db_note['version']
                        )
                        
                        if success:
                            updated_note = self.notes_service.get_note_by_id(user_id, db_note['id'])
                            if updated_note:
                                self._update_note_sync_metadata(
                                    db_note['id'], disk_file, root_path, user_id,
                                    updated_note['version']
                                )
                            progress.updated_notes.append(db_note['id'])
                            
                    elif db_changed and disk_changed:
                        # Both changed - CONFLICT!
                        conflict = SyncConflict(
                            note_id=db_note['id'],
                            file_path=rel_path,
                            conflict_type='both_changed',
                            db_content=db_note['content'],
                            disk_content=disk_file.content,
                            db_hash=db_content_hash,
                            disk_hash=disk_hash
                        )
                        progress.conflicts.append(conflict)
                        self._record_conflict(session_id, conflict)
                        
                        # Auto-resolve if not asking
                        if conflict_resolution == ConflictResolution.NEWER_WINS:
                            # Compare timestamps
                            db_modified = datetime.fromisoformat(db_note['last_modified'])
                            disk_modified = datetime.fromtimestamp(disk_file.mtime, tz=timezone.utc)
                            
                            if db_modified > disk_modified:
                                # DB is newer
                                disk_file.absolute_path.write_text(db_note['content'], encoding='utf-8')
                                progress.updated_files.append(disk_file.absolute_path)
                            else:
                                # Disk is newer
                                success = self.notes_service.update_note(
                                    user_id=user_id,
                                    note_id=db_note['id'],
                                    update_data={'content': disk_file.content},
                                    expected_version=db_note['version']
                                )
                                if success:
                                    progress.updated_notes.append(db_note['id'])
                                    
            except Exception as e:
                logger.error(f"Error syncing {rel_path}: {e}")
                progress.errors.append((str(rel_path), e))
            
            progress.processed_files += 1
            if self.progress_callback:
                self.progress_callback(progress)

#
# End of sync_engine.py
########################################################################################################################