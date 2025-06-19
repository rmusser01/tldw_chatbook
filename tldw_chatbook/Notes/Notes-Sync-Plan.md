# Notes Bi-Directional Sync Implementation Plan

## Overview

This document outlines the implementation of bi-directional file synchronization for the tldw_chatbook notes system, including the design decisions made during development and the rationale behind them.

## Project Goals

1. **Enable External Editor Integration**: Allow users to edit notes in their preferred editors (VSCode, Obsidian, etc.)
2. **Provide Backup Capability**: Maintain file-based copies of notes for safety
3. **Support Multiple Workflows**: Accommodate different user preferences for note management
4. **Ensure Data Integrity**: Prevent data loss through careful conflict detection and resolution
5. **Maintain Performance**: Handle large note collections efficiently

## Architecture Decisions

### 1. Layered Architecture

**Decision**: Implement three distinct layers:
- **Engine Layer** (`sync_engine.py`): Core sync logic
- **Service Layer** (`sync_service.py`): High-level operations and profile management
- **UI Layer** (`notes_sync_widget.py` + event handlers): User interface

**Rationale**:
- Separation of concerns for maintainability
- Engine can be tested independently
- Service layer provides clean API for UI
- Easy to add new UI implementations (CLI, web, etc.)

### 2. Database Schema Design

**Decision**: Add sync metadata directly to notes table rather than separate sync table

**Fields Added**:
```sql
file_path_on_disk         TEXT UNIQUE    -- Absolute path
relative_file_path_on_disk TEXT          -- Relative to sync root
sync_root_folder          TEXT           -- Root folder path
last_synced_disk_file_hash TEXT          -- SHA256 hash
last_synced_disk_file_mtime REAL         -- File modification time
is_externally_synced      BOOLEAN        -- Sync enabled flag
sync_strategy             TEXT           -- Direction preference
sync_excluded             BOOLEAN        -- Exclude from sync
file_extension            TEXT           -- File type (.md, .txt)
```

**Rationale**:
- Avoids complex joins for sync operations
- Maintains data locality
- Leverages existing version control system
- UNIQUE constraint on file_path prevents conflicts

### 3. Conflict Detection Strategy

**Decision**: Use content hashing (SHA256) as primary change detection mechanism

**Implementation**:
- Store hash of last synced content
- Compare current hashes with stored hash
- Detect three states: db_changed, disk_changed, both_changed

**Rationale**:
- More reliable than timestamps alone
- Handles clock skew between systems
- Detects actual content changes, not just touch events
- SHA256 provides sufficient uniqueness

### 4. Sync Direction Options

**Decision**: Support three sync modes:
1. **Disk → Database**: Import files as notes
2. **Database → Disk**: Export notes as files
3. **Bidirectional**: Full synchronization

**Rationale**:
- Covers all common use cases
- Allows gradual adoption (start with one-way)
- Supports both import and export workflows
- Bidirectional for power users

### 5. Conflict Resolution Strategies

**Decision**: Implement four resolution modes:
1. **Ask**: Interactive user choice
2. **Newer Wins**: Timestamp-based auto-resolution
3. **Database Wins**: Always prefer database
4. **Disk Wins**: Always prefer disk

**Rationale**:
- "Ask" preserves user control
- Auto modes enable unattended sync
- Simple strategies reduce complexity
- Covers most common preferences

### 6. Progress Tracking

**Decision**: Implement callback-based progress reporting

**Implementation**:
```python
def progress_callback(progress: SyncProgress):
    # Update UI with progress.processed_files, total_files, etc.
```

**Rationale**:
- Decouples engine from UI
- Enables real-time updates
- Supports different UI paradigms
- Allows progress persistence

### 7. Sync Profiles

**Decision**: Store reusable sync configurations as profiles

**Profile Contents**:
- Root folder path
- Sync direction
- Conflict resolution
- File extensions
- Auto-sync settings

**Rationale**:
- Reduces repetitive configuration
- Enables quick switching between projects
- Supports automation
- Simplifies UI

### 8. File Type Support

**Decision**: Default to .md and .txt, make extensible

**Rationale**:
- Covers most text note formats
- Markdown is standard for many tools
- Extensible for future formats
- Avoids binary file issues

## Implementation Phases

### Phase 1: Core Infrastructure ✅
- Database schema migration
- Sync engine implementation
- Basic service layer
- Notes library integration

### Phase 2: User Interface ✅
- Sync configuration widget
- Progress visualization
- Status indicators
- Event handlers

### Phase 3: Advanced Features ✅
- Sync profiles
- Conflict detection
- Auto-sync capability
- History tracking

### Phase 4: Testing & Documentation ✅
- Unit tests for sync engine
- Integration tests
- User documentation
- API reference

## Technical Decisions

### 1. Async Implementation

**Decision**: Use async/await for sync operations

**Rationale**:
- Non-blocking UI during sync
- Better resource utilization
- Natural fit with Textual framework
- Enables cancellation

### 2. Transaction Management

**Decision**: Wrap sync metadata updates in transactions

**Rationale**:
- Ensures consistency
- Enables rollback on error
- Prevents partial updates
- Maintains data integrity

### 3. Path Handling

**Decision**: Store both absolute and relative paths

**Rationale**:
- Absolute for direct file access
- Relative for portability
- Supports moving sync folders
- Enables path reconstruction

### 4. Error Handling

**Decision**: Collect errors without stopping sync

**Rationale**:
- Maximizes successful syncs
- Provides comprehensive error report
- Allows partial success
- Better user experience

### 5. Change Detection

**Decision**: Use modification time as quick check, hash for verification

**Rationale**:
- Fast initial scan
- Accurate change detection
- Handles edge cases
- Performance optimization

## Security Considerations

### 1. No Automatic Deletion

**Decision**: Never delete files/notes without explicit confirmation

**Rationale**:
- Prevents accidental data loss
- Builds user trust
- Allows recovery options
- Safety first approach

### 2. Conflict Preservation

**Decision**: Keep both versions during conflicts

**Rationale**:
- No data loss
- User chooses resolution
- Enables manual merge
- Preserves work

### 3. Path Validation

**Decision**: Validate all file paths before operations

**Rationale**:
- Prevents directory traversal
- Ensures sync boundary
- Security best practice
- Avoids system file access

## Performance Optimizations

### 1. Incremental Sync

**Decision**: Only process changed files

**Rationale**:
- Scales to large collections
- Reduces sync time
- Minimizes resource usage
- Better user experience

### 2. Batch Operations

**Decision**: Process multiple files in batches

**Rationale**:
- Reduces database round trips
- Improves throughput
- Configurable batch size
- Memory efficient

### 3. Index Usage

**Decision**: Add indexes for sync-related queries

**Rationale**:
- Fast file lookup
- Efficient filtering
- Quick status checks
- Scalability

## Future Enhancements

### Priority 1 (Next Release)
- [ ] Three-way merge for conflicts
- [ ] File watcher for real-time sync
- [ ] Diff viewer for conflicts
- [ ] Sync scheduling

### Priority 2 (Future)
- [ ] Cloud storage integration
- [ ] Git integration
- [ ] Multi-folder sync
- [ ] Sync templates

### Priority 3 (Long-term)
- [ ] Collaborative sync
- [ ] Encryption support
- [ ] Compression
- [ ] Selective sync

## Migration Strategy

### For Existing Users
1. Schema migration runs automatically
2. Existing notes remain unchanged
3. Sync is opt-in per note
4. No forced changes

### For New Users
1. Sync available immediately
2. Templates include sync-friendly formats
3. Guided setup available
4. Best practices documented

## Testing Strategy

### Unit Tests
- Sync engine logic
- Conflict detection
- Hash calculation
- File operations

### Integration Tests
- Full sync scenarios
- Profile management
- Error conditions
- Performance benchmarks

### User Acceptance Tests
- Common workflows
- Edge cases
- Error recovery
- Performance perception

## Documentation Requirements

### User Documentation
- Getting started guide
- Common workflows
- Troubleshooting
- Best practices

### Technical Documentation
- API reference
- Architecture overview
- Extension guide
- Development setup

## Success Metrics

### Functional
- ✅ Bidirectional sync works
- ✅ Conflicts detected accurately
- ✅ No data loss
- ✅ Profile management

### Performance
- ✅ 1000+ notes handled efficiently
- ✅ Sub-second incremental sync
- ✅ Minimal memory usage
- ✅ Responsive UI

### Usability
- ✅ Clear status indicators
- ✅ Intuitive configuration
- ✅ Helpful error messages
- ✅ Progress feedback

## Rollback Plan

If issues arise:
1. Sync can be disabled globally
2. Individual notes can be unlinked
3. Schema changes are backward compatible
4. Original functionality preserved

## Conclusion

The bi-directional sync feature has been designed with a focus on:
- **Safety**: No data loss, careful conflict handling
- **Flexibility**: Multiple sync modes and strategies
- **Performance**: Efficient for large note collections
- **Usability**: Clear UI and helpful feedback

The implementation provides a solid foundation that can be extended based on user feedback and emerging requirements.