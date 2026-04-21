# ADR-004: Voice Profile Management Design

**Date**: 2025-01-25  
**Status**: Accepted  
**Context**: Designing voice profile management system for Higgs Audio backend

## Decision

Implement a comprehensive voice profile management system with:
1. JSON-based profile storage with metadata
2. Import/export functionality for profile sharing
3. Audio validation and analysis capabilities
4. Backup and restore mechanisms
5. Tag-based organization

## Context

Higgs Audio's zero-shot voice cloning requires managing reference audio files and associated metadata. Users need to:
- Create and manage custom voice profiles
- Share voice profiles between systems
- Organize profiles with tags and descriptions
- Ensure audio quality and compatibility

## Rationale

### 1. Storage Architecture

**Decision**: JSON file with separate audio directories

**Structure**:
```
~/.config/tldw_cli/higgs_voices/
├── voice_profiles.json
├── backups/
│   └── voice_profiles_backup_*.json
├── profile_name_1/
│   └── reference.wav
└── profile_name_2/
    └── reference.mp3
```

**Reasons**:
- Human-readable profile metadata
- Easy backup and version control
- Organized audio file storage
- Simple migration between systems

### 2. Profile Schema

**Decision**: Comprehensive metadata structure

```json
{
  "profile_name": {
    "display_name": "Friendly Voice",
    "reference_audio": "/path/to/reference.wav",
    "language": "en",
    "description": "Warm, friendly voice",
    "tags": ["casual", "friendly"],
    "created_at": "2025-01-25T10:00:00",
    "updated_at": "2025-01-25T10:00:00",
    "metadata": {
      "mean_pitch_hz": 220.5,
      "estimated_tempo_bpm": 120.0
    },
    "audio_info": {
      "duration": 15.5,
      "sample_rate": 44100,
      "channels": 1
    }
  }
}
```

**Reasons**:
- Complete profile information
- Audio characteristics for matching
- Temporal tracking for updates
- Extensible metadata field

### 3. Audio Validation

**Decision**: Multi-level validation approach

**Checks**:
1. File existence and readability
2. Format validation (wav, mp3, flac, ogg, m4a)
3. Duration limits (max 5 minutes)
4. File size limits (max 100MB)
5. Audio integrity (if soundfile available)

**Reasons**:
- Prevents invalid profiles
- Protects against resource exhaustion
- Ensures compatibility with model
- Graceful degradation without dependencies

### 4. Import/Export System

**Decision**: Self-contained profile packages

**Export Format**:
```
higgs_voice_profilename/
├── profile.json
├── reference.wav
└── README.txt
```

**Features**:
- Complete profile portability
- Human-readable documentation
- Relative path references
- Optional overwrite on import

**Reasons**:
- Easy profile sharing
- No external dependencies
- Clear package structure
- Version migration support

### 5. Backup Strategy

**Decision**: Automatic rotating backups

**Implementation**:
- Backup before every save operation
- Timestamp-based naming
- Keep last 10 backups
- Simple restore mechanism

**Reasons**:
- Protection against corruption
- Easy rollback capability
- Limited disk usage
- No manual intervention needed

### 6. Audio Analysis

**Decision**: Optional librosa-based analysis

**Extracted Features**:
- Mean pitch and variance
- Estimated tempo
- Energy characteristics
- Spectral centroid

**Reasons**:
- Voice matching capabilities
- Quality assessment
- Optional dependency
- Scientific approach

## Consequences

### Positive
- Robust profile management
- Easy profile sharing
- Data integrity protection
- Rich metadata support
- Extensible design

### Negative
- Additional complexity vs simple file list
- Disk space for backups
- Optional dependencies for full features

### Neutral
- JSON format requires parsing
- Separate manager class needed
- Migration path for updates

## Usage Examples

### Creating a Profile
```python
manager = HiggsVoiceProfileManager(voice_dir)
success, msg = manager.create_profile(
    profile_name="narrator",
    reference_audio_path="/path/to/sample.wav",
    display_name="Story Narrator",
    description="Deep, engaging narrator voice",
    tags=["narration", "storytelling"]
)
```

### Exporting for Sharing
```python
success, msg = manager.export_profile(
    profile_name="narrator",
    export_path="/path/to/exports"
)
# Creates: /path/to/exports/higgs_voice_narrator/
```

### Importing Shared Profile
```python
success, msg = manager.import_profile(
    import_path="/downloads/higgs_voice_narrator",
    profile_name="imported_narrator",
    overwrite=False
)
```

## Security Considerations

1. **Path Validation**: All paths validated before operations
2. **Size Limits**: Prevent resource exhaustion
3. **No Code Execution**: JSON-only, no executable content
4. **Local Storage**: No network operations

## Future Enhancements

1. **Profile Versioning**: Track profile version history
2. **Cloud Sync**: Optional cloud backup/sync
3. **Profile Templates**: Pre-configured profile types
4. **Batch Operations**: Multiple profile import/export
5. **UI Integration**: Visual profile management

## References

- HiggsVoiceProfileManager implementation
- Voice profile schema design
- Backup rotation strategy