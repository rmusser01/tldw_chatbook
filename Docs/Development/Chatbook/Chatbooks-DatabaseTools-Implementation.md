# Database Tools & Chatbooks Implementation Summary

## Overview

This document summarizes the current database-tools and Chatbooks surfaces:
1. **Deprecated Tools Settings utility surface** - retained bulk database maintenance and advanced utilities
2. **Chatbooks System** - knowledge pack creation, import, and sharing functionality

## 1. Database Tools

### Retained `Tools_Settings_Window.py` utility surface

`Tools_Settings_Window.py` remains a deprecated utility and test surface; it is not the canonical settings or Chatbooks destination. Its database-tools area retains bulk operations across configured databases:

- Vacuum all configured databases
- Create a bulk backup of configured databases
- Check integrity across configured databases
- Run the remaining database-specific advanced utilities where available

Individual per-database vacuum, backup, restore, and integrity actions, including the former Last Backup labels, are no longer part of this legacy surface.

## 2. Chatbooks System

### Module Structure (`tldw_chatbook/Chatbooks/`)

#### Core Files
1. **`chatbook_models.py`** - Data structures
   - `ChatbookManifest` - Metadata and content listing
   - `ContentItem` - Individual content pieces
   - `Relationship` - Content relationships
   - `ChatbookContent` - Container for all content
   - `Chatbook` - Complete chatbook structure

2. **`chatbook_creator.py`** - Export functionality
   - `ChatbookCreator` class
   - Content collection from multiple databases
   - Relationship discovery
   - ZIP archive creation
   - Automatic dependency tracking

3. **`chatbook_importer.py`** - Import functionality
   - `ChatbookImporter` class
   - Preview functionality
   - Conflict resolution
   - Selective import
   - Progress tracking

4. **`conflict_resolver.py`** - Handle import conflicts
   - Multiple resolution strategies (Skip, Rename, Replace, Merge)
   - Content-specific conflict handling

### UI Implementation

`ChatbooksScreen` hosts `ChatbooksWindowImproved`, the canonical Chatbooks surface. It launches `ChatbookCreationWizard` for `Create Local` and `Create Server`, and `ChatbookImportWizard` for `Import Local` and `Import Server`.

`ChatbookCreationWindow.py` remains retained legacy utility support; it is not the canonical creation surface.

### Chatbook Format

```
chatbook_v1/
├── manifest.json          # Metadata and content listing
├── README.md             # Human-readable description
└── content/
    ├── conversations/    # JSON conversation files
    ├── notes/           # Markdown note files
    ├── characters/      # Character card JSON files
    ├── prompts/         # Prompt JSON files
    └── media/           # Media files (optional)
```

### Key Features

1. **Selective Content Export**
   - Choose specific conversations, notes, characters, etc.
   - Automatic dependency resolution
   - Configurable media inclusion

2. **Smart Import**
   - Preview before import
   - Conflict resolution options
   - Progress tracking
   - Prefix imported content

3. **Metadata & Versioning**
   - Version tracking for compatibility
   - Author attribution
   - Tags and categories
   - Creation/update timestamps

## Usage

### Creating a Chatbook
1. Open `ChatbooksScreen` / `ChatbooksWindowImproved`.
2. Choose `Create Local` or `Create Server`.
3. Complete the `ChatbookCreationWizard`.

### Importing a Chatbook
1. Open `ChatbooksScreen` / `ChatbooksWindowImproved`.
2. Choose `Import Local` or `Import Server`.
3. Complete the `ChatbookImportWizard`.

## Technical Notes

- All database operations run in background workers
- File operations use proper error handling
- Paths are validated and sanitized
- Temporary files cleaned up automatically
- Bulk database maintenance continues to use the shared backup and integrity infrastructure
