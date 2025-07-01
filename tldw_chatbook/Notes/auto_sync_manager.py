# auto_sync_manager.py
# Description: Manages automatic background synchronization for notes
#
# Imports
import asyncio
from pathlib import Path
from typing import Optional, Callable, Set
from datetime import datetime, timedelta
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = None
#
# Third-Party Imports
from loguru import logger
#
# Local Imports
from .sync_engine import SyncDirection, ConflictResolution
#
########################################################################################################################
#
# Classes:

class NotesFileWatcher(FileSystemEventHandler):
    """Watches for file changes in the notes directory."""
    
    def __init__(self, callback: Callable[[Path], None]):
        self.callback = callback
        self.changed_files: Set[Path] = set()
        self.last_event_time = datetime.now()
        
    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith('.md'):
            self.changed_files.add(Path(event.src_path))
            self.last_event_time = datetime.now()
            self.callback(Path(event.src_path))
    
    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith('.md'):
            self.changed_files.add(Path(event.src_path))
            self.last_event_time = datetime.now()
            self.callback(Path(event.src_path))
    
    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and event.src_path.endswith('.md'):
            # Remove from changed files if it was there
            self.changed_files.discard(Path(event.src_path))
            self.callback(Path(event.src_path))
    
    def get_changed_files(self) -> Set[Path]:
        """Get and clear the changed files list."""
        files = self.changed_files.copy()
        self.changed_files.clear()
        return files


class AutoSyncManager:
    """Manages automatic synchronization of notes."""
    
    def __init__(
        self,
        sync_service,
        sync_folder: Path,
        user_id: str,
        sync_interval: int = 300,  # 5 minutes default
        sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
        conflict_resolution: ConflictResolution = ConflictResolution.NEWER_WINS
    ):
        self.sync_service = sync_service
        self.sync_folder = sync_folder
        self.user_id = user_id
        self.sync_interval = sync_interval
        self.sync_direction = sync_direction
        self.conflict_resolution = conflict_resolution
        
        self.is_running = False
        self.sync_task: Optional[asyncio.Task] = None
        self.observer: Optional[Observer] = None
        self.file_watcher: Optional[NotesFileWatcher] = None
        
        self.last_sync_time = datetime.now()
        self.pending_sync = False
        self.sync_in_progress = False
        
        # Callbacks for UI updates
        self.on_sync_started: Optional[Callable] = None
        self.on_sync_completed: Optional[Callable] = None
        self.on_sync_error: Optional[Callable] = None
        self.on_files_changed: Optional[Callable] = None
    
    def start(self):
        """Start auto-sync monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start file watcher
        self._start_file_watcher()
        
        # Start sync loop
        self.sync_task = asyncio.create_task(self._sync_loop())
        
        logger.info(f"Auto-sync started for {self.sync_folder}")
    
    def stop(self):
        """Stop auto-sync monitoring."""
        self.is_running = False
        
        # Stop file watcher
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        # Cancel sync task
        if self.sync_task:
            self.sync_task.cancel()
            self.sync_task = None
        
        logger.info("Auto-sync stopped")
    
    def _start_file_watcher(self):
        """Start watching for file changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("Watchdog not available - file monitoring disabled")
            return
            
        if not self.sync_folder.exists():
            self.sync_folder.mkdir(parents=True, exist_ok=True)
        
        self.file_watcher = NotesFileWatcher(self._on_file_changed)
        self.observer = Observer()
        self.observer.schedule(
            self.file_watcher,
            str(self.sync_folder),
            recursive=True
        )
        self.observer.start()
    
    def _on_file_changed(self, file_path: Path):
        """Handle file change events."""
        # Mark that we need to sync
        self.pending_sync = True
        
        # Notify UI if callback is set
        if self.on_files_changed:
            asyncio.create_task(self._notify_files_changed())
    
    async def _notify_files_changed(self):
        """Notify UI about file changes."""
        if self.on_files_changed:
            changed_count = len(self.file_watcher.changed_files)
            self.on_files_changed(changed_count)
    
    async def _sync_loop(self):
        """Main sync loop that runs periodically."""
        while self.is_running:
            try:
                # Check if we need to sync
                should_sync = False
                
                # Sync if we have pending changes
                if self.pending_sync:
                    # Wait a bit for file operations to settle
                    time_since_last_event = datetime.now() - self.file_watcher.last_event_time
                    if time_since_last_event > timedelta(seconds=5):
                        should_sync = True
                
                # Sync if interval has passed
                time_since_last_sync = datetime.now() - self.last_sync_time
                if time_since_last_sync > timedelta(seconds=self.sync_interval):
                    should_sync = True
                
                if should_sync and not self.sync_in_progress:
                    await self._perform_sync()
                
                # Wait before next check
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                if self.on_sync_error:
                    self.on_sync_error(str(e))
                
                # Wait longer on error
                await asyncio.sleep(30)
    
    async def _perform_sync(self):
        """Perform the actual sync operation."""
        self.sync_in_progress = True
        self.pending_sync = False
        
        try:
            # Notify sync started
            if self.on_sync_started:
                self.on_sync_started()
            
            # Get changed files
            changed_files = self.file_watcher.get_changed_files() if self.file_watcher else set()
            
            # Perform sync
            session_id = await self.sync_service.sync(
                user_id=self.user_id,
                sync_root=self.sync_folder,
                direction=self.sync_direction,
                conflict_resolution=self.conflict_resolution
            )
            
            # Get results
            results = self.sync_service.get_session_results(session_id)
            
            # Update last sync time
            self.last_sync_time = datetime.now()
            
            # Notify sync completed
            if self.on_sync_completed:
                self.on_sync_completed(results)
            
            logger.debug(f"Auto-sync completed: {results}")
            
        except Exception as e:
            logger.error(f"Auto-sync failed: {e}")
            if self.on_sync_error:
                self.on_sync_error(str(e))
        
        finally:
            self.sync_in_progress = False
    
    async def trigger_sync(self):
        """Manually trigger a sync operation."""
        if not self.sync_in_progress:
            await self._perform_sync()
    
    def update_settings(
        self,
        sync_interval: Optional[int] = None,
        sync_direction: Optional[SyncDirection] = None,
        conflict_resolution: Optional[ConflictResolution] = None
    ):
        """Update auto-sync settings."""
        if sync_interval is not None:
            self.sync_interval = sync_interval
        if sync_direction is not None:
            self.sync_direction = sync_direction
        if conflict_resolution is not None:
            self.conflict_resolution = conflict_resolution
        
        logger.info("Auto-sync settings updated")

#
# End of auto_sync_manager.py
########################################################################################################################