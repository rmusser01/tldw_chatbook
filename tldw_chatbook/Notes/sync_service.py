# sync_service.py
# Description: Service layer for note synchronization operations
#
# Imports
import asyncio
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
    NotesSyncEngine, SyncDirection, ConflictResolution, 
    SyncProgress, SyncConflict, SyncStatus
)
from .Notes_Library import NotesInteropService
from ..DB.ChaChaNotes_DB import CharactersRAGDB
#
########################################################################################################################
#
# Classes:

class SyncProfile:
    """Represents a saved sync configuration."""
    
    def __init__(self, name: str, root_folder: Path, direction: SyncDirection,
                 conflict_resolution: ConflictResolution, extensions: List[str] = None,
                 auto_sync: bool = False, sync_interval: int = 300):
        self.name = name
        self.root_folder = root_folder
        self.direction = direction
        self.conflict_resolution = conflict_resolution
        self.extensions = extensions or ['.md', '.txt']
        self.auto_sync = auto_sync
        self.sync_interval = sync_interval  # seconds
        self.last_sync: Optional[datetime] = None
        self.last_session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            'name': self.name,
            'root_folder': str(self.root_folder),
            'direction': self.direction.value,
            'conflict_resolution': self.conflict_resolution.value,
            'extensions': self.extensions,
            'auto_sync': self.auto_sync,
            'sync_interval': self.sync_interval,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'last_session_id': self.last_session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncProfile':
        """Create profile from dictionary."""
        profile = cls(
            name=data['name'],
            root_folder=Path(data['root_folder']),
            direction=SyncDirection(data['direction']),
            conflict_resolution=ConflictResolution(data['conflict_resolution']),
            extensions=data.get('extensions', ['.md', '.txt']),
            auto_sync=data.get('auto_sync', False),
            sync_interval=data.get('sync_interval', 300)
        )
        
        if data.get('last_sync'):
            profile.last_sync = datetime.fromisoformat(data['last_sync'])
        profile.last_session_id = data.get('last_session_id')
        
        return profile


class NotesSyncService:
    """High-level service for note synchronization."""
    
    def __init__(self, notes_service: NotesInteropService, db: CharactersRAGDB,
                 config_path: Optional[Path] = None):
        """
        Initialize sync service.
        
        Args:
            notes_service: Notes service for database operations
            db: Database instance
            config_path: Path to store sync profiles and configuration
        """
        self.notes_service = notes_service
        self.db = db
        self.config_path = config_path or Path.home() / '.config' / 'tldw_cli' / 'sync_profiles.json'
        self.sync_engine = NotesSyncEngine(notes_service, db)
        self.profiles: Dict[str, SyncProfile] = {}
        self._auto_sync_tasks: Dict[str, asyncio.Task] = {}
        self._load_profiles()
    
    def _load_profiles(self):
        """Load sync profiles from configuration file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    for profile_data in data.get('profiles', []):
                        profile = SyncProfile.from_dict(profile_data)
                        self.profiles[profile.name] = profile
                logger.info(f"Loaded {len(self.profiles)} sync profiles")
            except Exception as e:
                logger.error(f"Error loading sync profiles: {e}")
    
    def _save_profiles(self):
        """Save sync profiles to configuration file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            profiles_data = [profile.to_dict() for profile in self.profiles.values()]
            
            with open(self.config_path, 'w') as f:
                json.dump({'profiles': profiles_data}, f, indent=2)
            
            logger.info(f"Saved {len(self.profiles)} sync profiles")
        except Exception as e:
            logger.error(f"Error saving sync profiles: {e}")
    
    def create_profile(self, name: str, root_folder: Path, direction: SyncDirection,
                      conflict_resolution: ConflictResolution, extensions: List[str] = None,
                      auto_sync: bool = False, sync_interval: int = 300) -> SyncProfile:
        """Create and save a new sync profile."""
        profile = SyncProfile(
            name=name,
            root_folder=root_folder,
            direction=direction,
            conflict_resolution=conflict_resolution,
            extensions=extensions,
            auto_sync=auto_sync,
            sync_interval=sync_interval
        )
        
        self.profiles[name] = profile
        self._save_profiles()
        
        if auto_sync:
            self.start_auto_sync(name)
        
        return profile
    
    def delete_profile(self, name: str) -> bool:
        """Delete a sync profile."""
        if name in self.profiles:
            # Stop auto-sync if running
            self.stop_auto_sync(name)
            
            del self.profiles[name]
            self._save_profiles()
            return True
        return False
    
    def get_profile(self, name: str) -> Optional[SyncProfile]:
        """Get a sync profile by name."""
        return self.profiles.get(name)
    
    def list_profiles(self) -> List[SyncProfile]:
        """List all sync profiles."""
        return list(self.profiles.values())
    
    async def sync_with_profile(self, profile_name: str, user_id: str,
                               progress_callback: Optional[Callable[[SyncProgress], None]] = None) -> Tuple[str, SyncProgress]:
        """Execute sync using a saved profile."""
        profile = self.profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        
        # Update sync engine with progress callback
        self.sync_engine.progress_callback = progress_callback
        
        # Execute sync
        session_id, progress = await self.sync_engine.sync(
            root_path=profile.root_folder,
            user_id=user_id,
            direction=profile.direction,
            conflict_resolution=profile.conflict_resolution,
            extensions=profile.extensions
        )
        
        # Update profile
        profile.last_sync = datetime.now()
        profile.last_session_id = session_id
        self._save_profiles()
        
        return session_id, progress
    
    async def sync_folder(self, root_folder: Path, user_id: str,
                         direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
                         conflict_resolution: ConflictResolution = ConflictResolution.ASK,
                         extensions: List[str] = None,
                         progress_callback: Optional[Callable[[SyncProgress], None]] = None) -> Tuple[str, SyncProgress]:
        """Execute one-time sync for a folder."""
        self.sync_engine.progress_callback = progress_callback
        
        return await self.sync_engine.sync(
            root_path=root_folder,
            user_id=user_id,
            direction=direction,
            conflict_resolution=conflict_resolution,
            extensions=extensions
        )
    
    def get_sync_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent sync session history."""
        history = []
        
        with self.db.transaction() as conn:
            cursor = conn.execute("""
                SELECT session_id, sync_root_folder, sync_direction, 
                       conflict_resolution, started_at, completed_at,
                       status, total_files, processed_files, conflicts_found,
                       errors_count, summary
                FROM sync_sessions
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,))
            
            for row in cursor:
                session_data = {
                    'session_id': row[0],
                    'sync_root_folder': row[1],
                    'sync_direction': row[2],
                    'conflict_resolution': row[3],
                    'started_at': row[4],
                    'completed_at': row[5],
                    'status': row[6],
                    'total_files': row[7],
                    'processed_files': row[8],
                    'conflicts_found': row[9],
                    'errors_count': row[10],
                    'summary': json.loads(row[11]) if row[11] else None
                }
                history.append(session_data)
        
        return history
    
    def get_conflicts_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conflicts for a specific sync session."""
        conflicts = []
        
        with self.db.transaction() as conn:
            cursor = conn.execute("""
                SELECT id, note_id, file_path, conflict_type,
                       db_content_hash, disk_content_hash,
                       db_modified_time, disk_modified_time,
                       resolution, resolved_at, created_at
                FROM sync_conflicts
                WHERE session_id = ?
                ORDER BY created_at DESC
            """, (session_id,))
            
            for row in cursor:
                conflict_data = {
                    'id': row[0],
                    'note_id': row[1],
                    'file_path': row[2],
                    'conflict_type': row[3],
                    'db_content_hash': row[4],
                    'disk_content_hash': row[5],
                    'db_modified_time': row[6],
                    'disk_modified_time': row[7],
                    'resolution': row[8],
                    'resolved_at': row[9],
                    'created_at': row[10]
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
                cursor = conn.execute("""
                    SELECT note_id, file_path, conflict_type, session_id
                    FROM sync_conflicts
                    WHERE id = ? AND resolution IS NULL
                """, (conflict_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Conflict {conflict_id} not found or already resolved")
                    return False
                
                note_id, file_path, conflict_type, session_id = row
                
                # Update conflict resolution
                conn.execute("""
                    UPDATE sync_conflicts
                    SET resolution = ?, resolved_at = ?
                    WHERE id = ?
                """, (resolution, datetime.now().isoformat(), conflict_id))
                
                # TODO: Implement actual resolution logic based on resolution type
                # This would involve updating the note or file based on the resolution
                
                return True
                
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_id}: {e}")
            return False
    
    def get_notes_sync_status(self, root_folder: Optional[Path] = None) -> List[Dict[str, Any]]:
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
                    'id': row[0],
                    'title': row[1],
                    'file_path': row[2],
                    'sync_root_folder': row[3],
                    'last_synced_hash': row[4],
                    'last_synced_mtime': row[5],
                    'is_synced': row[6],
                    'sync_strategy': row[7],
                    'sync_excluded': row[8],
                    'last_modified': row[9],
                    'current_hash': self.sync_engine._calculate_hash(row[10])
                }
                
                # Check sync status
                if note_data['file_path'] and Path(note_data['file_path']).exists():
                    try:
                        file_content = Path(note_data['file_path']).read_text(encoding='utf-8')
                        file_hash = self.sync_engine._calculate_hash(file_content)
                        
                        if file_hash != note_data['last_synced_hash']:
                            if note_data['current_hash'] != note_data['last_synced_hash']:
                                note_data['sync_status'] = 'conflict'
                            else:
                                note_data['sync_status'] = 'file_changed'
                        elif note_data['current_hash'] != note_data['last_synced_hash']:
                            note_data['sync_status'] = 'db_changed'
                        else:
                            note_data['sync_status'] = 'synced'
                    except (OSError, IOError, UnicodeDecodeError) as e:
                        logger.warning(f"Error reading file {note_data['file_path']}: {e}")
                        note_data['sync_status'] = 'file_error'
                else:
                    note_data['sync_status'] = 'file_missing'
                
                notes_status.append(note_data)
        
        return notes_status
    
    def start_auto_sync(self, profile_name: str):
        """Start auto-sync for a profile."""
        profile = self.profiles.get(profile_name)
        if not profile or not profile.auto_sync:
            return
        
        if profile_name in self._auto_sync_tasks:
            logger.warning(f"Auto-sync already running for profile '{profile_name}'")
            return
        
        async def auto_sync_loop():
            """Auto-sync loop for a profile."""
            while profile_name in self._auto_sync_tasks:
                try:
                    # TODO: Get user_id from session or config
                    user_id = "auto_sync"  # Placeholder
                    
                    logger.info(f"Running auto-sync for profile '{profile_name}'")
                    await self.sync_with_profile(profile_name, user_id)
                    
                    # Wait for next sync
                    await asyncio.sleep(profile.sync_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in auto-sync for profile '{profile_name}': {e}")
                    await asyncio.sleep(60)  # Wait before retry
        
        task = asyncio.create_task(auto_sync_loop())
        self._auto_sync_tasks[profile_name] = task
        logger.info(f"Started auto-sync for profile '{profile_name}'")
    
    def stop_auto_sync(self, profile_name: str):
        """Stop auto-sync for a profile."""
        if profile_name in self._auto_sync_tasks:
            task = self._auto_sync_tasks.pop(profile_name)
            task.cancel()
            logger.info(f"Stopped auto-sync for profile '{profile_name}'")
    
    def stop_all_auto_syncs(self):
        """Stop all auto-sync tasks."""
        for profile_name in list(self._auto_sync_tasks.keys()):
            self.stop_auto_sync(profile_name)

#
# End of sync_service.py
########################################################################################################################