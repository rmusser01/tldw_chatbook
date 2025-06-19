# Notes Bi-Directional Sync Feature

## Overview

The bi-directional sync feature allows you to synchronize your notes between the tldw_chatbook database and files on your local filesystem. This enables you to:

- Edit notes in your favorite external editor (VSCode, Obsidian, etc.)
- Maintain a file-based backup of your notes
- Integrate with other note-taking systems
- Work with notes offline and sync changes later

## Key Features

### 1. Flexible Sync Directions
- **Disk → Database**: Import files as notes
- **Database → Disk**: Export notes as files
- **Bidirectional**: Keep notes and files in sync

### 2. Smart Conflict Resolution
- **Ask on Conflict**: Prompt user to choose which version to keep
- **Newer Wins**: Automatically use the most recently modified version
- **Database Wins**: Always prefer the database version
- **Disk Wins**: Always prefer the file version

### 3. Sync Profiles
Save your sync configurations as profiles for quick access:
- Root folder location
- Sync direction preference
- Conflict resolution strategy
- File extensions to include
- Auto-sync settings

### 4. Visual Sync Status
Notes display sync status indicators:
- ✓ Synced (up to date)
- ↓ File changed (needs sync from disk)
- ↑ Database changed (needs sync to disk)
- ⚠ Conflict (both changed)
- ✗ File missing
- ! File error

### 5. Progress Tracking
Real-time progress updates during sync operations:
- Files processed counter
- Progress bar
- Conflict notifications
- Error reporting

## Database Schema

The following fields are added to the notes table for sync support:

```sql
-- File sync metadata
file_path_on_disk TEXT UNIQUE,           -- Absolute path to synced file
relative_file_path_on_disk TEXT,         -- Path relative to sync root
sync_root_folder TEXT,                   -- Root folder for this sync
last_synced_disk_file_hash TEXT,         -- SHA256 hash at last sync
last_synced_disk_file_mtime REAL,        -- File modification time at last sync
is_externally_synced BOOLEAN DEFAULT 0,  -- Whether note is synced
sync_strategy TEXT,                      -- Preferred sync direction
sync_excluded BOOLEAN DEFAULT 0,         -- Exclude from sync
file_extension TEXT DEFAULT '.md'        -- File extension to use
```

## Architecture

### Components

1. **NotesSyncEngine** (`sync_engine.py`)
   - Core sync logic
   - File scanning and comparison
   - Conflict detection
   - Hash calculation
   - Progress tracking

2. **NotesSyncService** (`sync_service.py`)
   - High-level sync operations
   - Profile management
   - Auto-sync scheduling
   - History tracking
   - Conflict resolution

3. **NotesSyncWidget** (`notes_sync_widget.py`)
   - UI for sync configuration
   - Profile management interface
   - Progress visualization
   - Status indicators

4. **Sync Event Handlers** (`notes_sync_events.py`)
   - Handle UI interactions
   - Trigger sync operations
   - Update displays

## Usage

### Quick Sync

1. Navigate to the Notes tab
2. Open the Sync panel
3. Select a folder to sync
4. Choose sync direction and conflict resolution
5. Click "Start Sync"

### Creating a Sync Profile

1. Configure sync settings as desired
2. Click "Save as Profile"
3. Enter a profile name
4. Profile is saved for future use

### Using Sync Profiles

1. Select a profile from the list
2. Click "Sync Selected"
3. Sync runs with saved settings

### Auto-Sync

1. Edit a sync profile
2. Enable "Auto-sync"
3. Set sync interval (default: 5 minutes)
4. Auto-sync runs in background

## Sync Algorithm

### Disk to Database
1. Scan directory for supported files (.md, .txt)
2. For each file:
   - If new: Create note in database
   - If changed: Update note content
   - If deleted: Unlink from database

### Database to Disk
1. Query synced notes for folder
2. For each note:
   - If new: Create file on disk
   - If changed: Update file content
   - If file missing: Create or unlink

### Bidirectional
1. Compare all files and notes
2. Detect changes using content hash
3. Identify conflicts (both changed)
4. Apply conflict resolution strategy
5. Update metadata after sync

## Conflict Resolution

### Detection
Conflicts occur when both file and database have changed since last sync:
- Compare current hashes with last synced hash
- Check modification timestamps
- Flag items that changed on both sides

### Resolution Strategies
1. **Ask**: Show conflict dialog with options
2. **Newer Wins**: Compare timestamps, use newer
3. **Database Wins**: Always use database version
4. **Disk Wins**: Always use file version

## Performance Considerations

### Optimizations
- Incremental sync using modification times
- Parallel file processing
- Hash-based change detection
- Configurable batch sizes
- Progress callbacks for UI updates

### Scalability
- Handles thousands of notes efficiently
- Minimal memory usage with streaming
- Database indexes for fast queries
- Async operations prevent UI blocking

## Security & Safety

### Data Protection
- No files deleted without confirmation
- Original content preserved in conflicts
- Sync history tracked in database
- Version control via optimistic locking

### Error Handling
- Graceful handling of file permissions
- Network interruption recovery
- Transaction rollback on failure
- Detailed error logging

## Future Enhancements

### Planned Features
1. **Three-way merge** for conflict resolution
2. **File watcher** for real-time sync
3. **Sync filters** by tags/keywords
4. **Cloud storage** integration
5. **Diff viewer** for conflicts
6. **Backup/restore** functionality
7. **Sync scheduling** with cron syntax
8. **Multi-folder** sync support

### Integration Possibilities
- Git integration for version control
- Obsidian vault compatibility
- Notion export/import
- Markdown preview during sync
- Template-based file creation

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Ensure read/write access to sync folder
   - Check file ownership

2. **Sync Conflicts**
   - Review conflict resolution settings
   - Use sync history to understand changes

3. **Performance Issues**
   - Reduce sync frequency
   - Limit file extensions
   - Use incremental sync

### Debug Mode
Enable debug logging for detailed sync information:
```python
logger.setLevel(logging.DEBUG)
```

## API Reference

### NotesInteropService Methods

```python
# Get notes configured for sync
get_notes_for_sync(user_id: str, sync_root_folder: Path = None) -> List[Dict]

# Get unsynced notes
get_unsynced_notes(user_id: str, limit: int = 100) -> List[Dict]

# Update sync metadata
update_note_sync_metadata(user_id: str, note_id: str, 
                         sync_metadata: Dict, expected_version: int) -> bool

# Link note to file
link_note_to_file(user_id: str, note_id: str, file_path: Path,
                  sync_root_folder: Path, sync_strategy: str) -> bool

# Unlink note from file
unlink_note_from_file(user_id: str, note_id: str) -> bool
```

### NotesSyncEngine Methods

```python
# Main sync method
async sync(root_path: Path, user_id: str, 
          direction: SyncDirection, 
          conflict_resolution: ConflictResolution,
          extensions: List[str] = None) -> Tuple[str, SyncProgress]

# Cancel sync
cancel_sync(session_id: str)

# Check cancellation
is_cancelled(session_id: str) -> bool
```

## Configuration

### Sync Settings
Default configuration stored in `~/.config/tldw_cli/sync_profiles.json`:

```json
{
  "profiles": [
    {
      "name": "Work Notes",
      "root_folder": "/path/to/notes",
      "direction": "bidirectional",
      "conflict_resolution": "newer_wins",
      "extensions": [".md", ".txt"],
      "auto_sync": true,
      "sync_interval": 300
    }
  ]
}
```

### File Extensions
Supported by default:
- `.md` - Markdown files
- `.txt` - Plain text files

Add custom extensions in sync profile configuration.