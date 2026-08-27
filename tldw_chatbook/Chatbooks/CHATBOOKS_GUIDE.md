# Chatbooks Guide

## Overview

Chatbooks are knowledge packs that allow you to export, share, and import curated collections of content from tldw_chatbook. They provide a way to package conversations, notes, characters, media, prompts, and kept briefings (with their cast scripts) into a portable format.

## Features

### 1. **Chatbook Creation**
- Multi-step wizard interface for guided creation
- Smart content selection with search and filtering
- Flexible export options (compression, formats, privacy)
- Dependency tracking for characters
- Progress tracking with detailed status

### 2. **Chatbook Import**
- Preview chatbooks before importing
- Validation to ensure compatibility
- Conflict resolution strategies:
  - Skip existing items
  - Rename imported items
  - Replace existing items
  - Merge content intelligently
- Import options for media, embeddings, and timestamps

### 3. **Export Management**
- View all exported chatbooks
- Re-export with different settings
- Delete old exports
- Share chatbooks (email, cloud - coming soon)
- Open chatbook locations

### 4. **Templates System**
- Pre-configured templates for common use cases:
  - Research Project
  - Creative Writing
  - Learning Journey
  - Project Documentation
  - Personal Assistant
  - Knowledge Base

## Creating a Chatbook

### Using the Wizard

1. Click "Start Creation Wizard" on the Chatbooks tab
2. **Step 1: Basic Information**
   - Enter chatbook name (required)
   - Add description
   - Include tags for categorization
   - Specify author name

3. **Step 2: Content Selection**
   - Use the smart content tree to select items
   - Search and filter content
   - Bulk selection operations (Select All, None, Invert)
   - Visual indicators for selected items

4. **Step 3: Export Options**
   - Choose format (ZIP, JSON, SQLite, Markdown)
   - Enable compression
   - Select what to include:
     - Embeddings (for RAG search)
     - Media files
     - Metadata and timestamps
     - User preferences
   - Privacy options:
     - Anonymize user names
     - Remove sensitive data
     - Include license file

5. **Step 4: Preview & Confirm**
   - Review chatbook details
   - Preview file structure
   - Confirm export location
   - Change location if needed

6. **Step 5: Progress & Completion**
   - Watch real-time progress
   - View status of each operation
   - Open folder when complete
   - Create another chatbook

### Using Templates

1. Click "Browse Templates"
2. Select a template that matches your use case
3. The creation wizard will be pre-configured
4. Customize as needed

## Importing a Chatbook

1. Click "Import Chatbook" on the Chatbooks tab
2. **Step 1: File Selection**
   - Browse for .zip chatbook file
   - Or drag & drop (coming soon)

3. **Step 2: Preview & Validation**
   - View chatbook metadata
   - Check content summary
   - Validate compatibility

4. **Step 3: Conflict Resolution**
   - Choose how to handle existing items
   - Preview potential conflicts

5. **Step 4: Import Options**
   - Select what to import
   - Configure tag handling
   - Enable backup creation

6. **Step 5: Import Progress**
   - Monitor import status
   - View summary statistics

## Managing Exports

Access the Export Management window to:

- **View Details**: See metadata, content summary, and statistics
- **Re-export**: Create new versions with different settings
- **Delete**: Remove old or unwanted exports
- **Share**: Email or upload to cloud (coming soon)
- **Open Location**: Access chatbook files directly

## Chatbook Format

### Structure
```
chatbook.zip
├── manifest.json          # Metadata and content listing
├── README.md             # Human-readable description
├── content/
│   ├── conversations/    # Exported conversations
│   ├── notes/           # Exported notes
│   ├── characters/      # Character profiles
│   ├── media/          # Media files (optional)
│   ├── prompts/        # Custom prompts
│   └── kept_briefings/ # Kept briefings (JSON + Markdown; scripts nested inside)
└── metadata/
    ├── relationships.json  # Content relationships
    └── embeddings/        # Vector embeddings (optional)
```

### Kept Briefings

A kept briefing (Watchlists/briefings you chose to keep, plus any cast
scripts made from it) exports as one JSON file (the machine-round-trippable
source of truth, including every provenance column) and one companion
Markdown file (a human-readable rendition) per briefing under
`content/kept_briefings/`. Kept scripts are not independently selectable in
the content picker -- they always travel with their parent kept briefing,
nested inside its JSON payload.

Kept briefings/scripts are local-only for ChaChaNotes sync between devices
(a deliberate v1 decision -- see
`Docs/superpowers/specs/2026-08-01-kept-briefings-design.md`); chatbook
export/import is the supported way to carry them to another device. On
import, a kept briefing whose `source_briefing_id` already exists locally
with different content is never silently overwritten -- it is reported as a
conflict in the import summary, and re-importing the same chatbook never
creates duplicates.

### Conversation thinking and replay policy

Chatbook V2 conversation entries preserve supported model-thinking envelopes
and the conversation's Auto/Include/Exclude thinking-history policy so an
imported conversation can hydrate the same collapsed Thinking activity and
future replay preference. Selected-conversation JSON provides the same
machine-round-trippable fields. These importable formats carry a sensitive-data
warning whenever displayable thinking or private provider continuation is
present.

Treat those files as sensitive even when the visible answer looks harmless.
Ordinary human-readable text and Markdown exports omit model thinking, private
continuation, and the proprietary UI notice. Chatbook also does not add
thinking to search, summaries, titles, logs, errors, speech, usage displays, or
diagnostic formats. The feature records only actual adapter-reported evidence;
it does not claim to recover hidden chain-of-thought.

Imports reject malformed or unsupported future thinking versions before
mutating the destination. A successfully imported supported envelope remains
owned by its assistant generation; editing, replacing, or deleting that
generation clears its thinking and separately protected continuation together.

### Manifest Schema
The manifest.json file contains:
- Version information
- Chatbook metadata (name, author, dates)
- Content inventory with IDs and types
- Relationships between content items
- Configuration settings
- Statistics

## Error Handling

The chatbooks feature includes comprehensive error handling:

- **File Errors**: Missing files, invalid formats
- **Permission Errors**: Access denied issues
- **Space Errors**: Insufficient disk space
- **Validation Errors**: Invalid content or data
- **Import Conflicts**: Duplicate content handling

Each error provides:
- Clear error message
- Detailed information
- Recovery suggestions
- Logging for debugging

## Best Practices

1. **Organization**
   - Use descriptive names for chatbooks
   - Add tags for easy categorization
   - Include detailed descriptions

2. **Content Selection**
   - Review dependencies before export
   - Use search to find specific content
   - Consider file size when including media

3. **Sharing**
   - Verify no sensitive data is included
   - Use anonymization when appropriate
   - Include license information

4. **Storage**
   - Keep chatbooks organized in folders
   - Regular cleanup of old exports
   - Backup important chatbooks

## Troubleshooting

### Common Issues

**Cannot create chatbook**
- Check disk space
- Verify write permissions
- Ensure selected content exists

**Import fails**
- Verify chatbook file is valid
- Check version compatibility
- Review conflict resolution settings

**Missing content after import**
- Check conflict resolution strategy
- Verify content was included in export
- Review import options

### Getting Help

1. Check error messages and suggestions
2. Review application logs
3. Consult this guide
4. Report issues on GitHub

## Future Enhancements

Planned features include:
- Cloud storage integration
- Collaborative chatbooks
- Version control
- Automated backups
- Enhanced sharing options
- Custom export formats
