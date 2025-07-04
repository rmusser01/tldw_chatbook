# Notes Import System Documentation

## Overview

The Notes Import System in tldw_chatbook provides a flexible way to import notes from various file formats. You can import content as either regular notes or note templates, supporting plain text files, structured data formats, and more.

## Table of Contents

1. [Supported File Formats](#supported-file-formats)
2. [Import Types](#import-types)
3. [User Guide](#user-guide)
4. [File Format Specifications](#file-format-specifications)
5. [Template System](#template-system)
6. [Developer Guide](#developer-guide)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Supported File Formats

The Notes Import System supports the following file formats:

### Plain Text Formats
- **Text Files** (`.txt`, `.text`) - Simple text files where filename becomes the note title
- **Markdown** (`.md`, `.markdown`) - Markdown files with optional title extraction from first heading
- **reStructuredText** (`.rst`) - ReStructuredText documentation files

### Structured Data Formats
- **JSON** (`.json`) - Structured notes with metadata support
- **YAML** (`.yaml`, `.yml`) - Human-readable structured format
- **CSV** (`.csv`) - Tabular data where each row becomes a note

## Import Types

### Import as Notes
Regular notes are imported directly into your notes database. They appear immediately in your Notes list and can be edited, synced, and managed like any other note.

### Import as Templates
Templates are imported into your template library. They become available in the "Create from Template" dropdown when creating new notes. Templates are stored in `~/.config/tldw_cli/note_templates.json`.

**Important**: After importing templates, you need to restart the application for them to appear in the template dropdown.

## User Guide

### Basic Import Process

1. Navigate to the **Ingest** tab
2. Click on **Notes** in the sidebar
3. Choose your import type:
   - Select "Import as Notes" for regular notes
   - Select "Import as Templates" for note templates
4. Click **Select Notes File(s)**
5. Choose your file(s) from the file picker
6. Review the preview showing how your notes will be imported
7. Click **Import Selected Notes Now**

### Import Type Indicators

In the preview, notes are marked with indicators:
- `[TEMPLATE]` prefix - File will be imported as a template
- `[template-name]` prefix - Note uses a specific template
- No prefix - Regular note

## File Format Specifications

### Plain Text Files

For `.txt`, `.md`, and `.rst` files:
- **Title**: Extracted from filename (without extension)
- **Content**: Entire file content
- **Special handling for Markdown**: If the file starts with a `# Heading`, that heading becomes the title

Example:
```markdown
# My Important Note
This is the content of my note...
```

### JSON Format

JSON files can contain single notes or arrays of notes:

#### Single Note
```json
{
    "title": "Meeting Notes",
    "content": "Discussion points from today's meeting...",
    "template": "meeting",
    "keywords": ["meeting", "project-x"],
    "created_at": "2024-01-15T10:30:00Z"
}
```

#### Multiple Notes
```json
[
    {
        "title": "Note 1",
        "content": "Content of first note"
    },
    {
        "title": "Note 2",
        "content": "Content of second note",
        "keywords": ["todo", "urgent"]
    }
]
```

#### Template Definition
```json
{
    "title": "Meeting Template",
    "content": "## Attendees\n\n## Agenda\n\n## Action Items\n\n## Next Steps",
    "is_template": true
}
```

### YAML Format

YAML files follow the same structure as JSON but in YAML syntax:

```yaml
title: Project Planning
content: |
  ## Goals
  - Define project scope
  - Set milestones
  
  ## Timeline
  Q1 2024 - Initial development
template: project
keywords:
  - planning
  - project-management
```

### CSV Format

CSV files should have columns for note data:

```csv
title,content,keywords
"Daily Standup","What I did yesterday...\nWhat I'm doing today...","standup,daily"
"Sprint Review","Sprint accomplishments...","sprint,review"
```

If columns aren't named `title` and `content`, the first two columns are used.

## Template System

### Creating Templates

Templates can be created in two ways:

1. **Import as Template**: Any supported file can be imported as a template
2. **JSON Template Format**: Use `"is_template": true` in JSON files

### Template Structure

Templates support placeholder variables and structured content:

```json
{
    "title": "Code Review Template",
    "content": "## Code Review\n\n**PR/Branch**: \n**Author**: \n**Reviewer**: {{user}}\n**Date**: {{date}}\n\n### Summary\n\n### Issues Found\n\n### Suggestions\n\n### Approval Status\n- [ ] Approved\n- [ ] Needs Changes\n- [ ] Rejected",
    "is_template": true
}
```

### Using Templates

After importing templates and restarting the app:
1. Go to Notes tab
2. Click "Create from Template"
3. Select your imported template
4. Fill in the placeholders

## Developer Guide

### File Handler Architecture

The import system uses a plugin-based architecture:

```python
from tldw_chatbook.Utils.note_importers import note_importer_registry

# Parse a file
notes = note_importer_registry.parse_file(
    file_path="notes.json",
    import_as_template=False
)

# Each note is a ParsedNote object
for note in notes:
    print(f"Title: {note.title}")
    print(f"Content: {note.content}")
    print(f"Is Template: {note.is_template}")
```

### Adding New File Formats

To support a new file format, create a handler:

```python
from tldw_chatbook.Utils.note_importers import NoteImporter, ParsedNote

class MyFormatImporter(NoteImporter):
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.myformat'
    
    def parse_file(self, file_path: Path) -> List[ParsedNote]:
        # Parse your format and return ParsedNote objects
        pass
```

### Metadata Support

All formats support extended metadata:

```python
ParsedNote(
    title="My Note",
    content="Content here",
    template="template_name",  # Template to use
    keywords=["tag1", "tag2"], # Tags/keywords
    created_at=datetime.now(), # Creation timestamp
    updated_at=datetime.now(), # Update timestamp
    metadata={"custom": "data"}, # Any additional data
    is_template=False  # Import as template?
)
```

## Examples

### Example: Import Meeting Notes

**meeting_notes.json**:
```json
[
    {
        "title": "Team Standup - Jan 15",
        "content": "## Attendees\n- Alice\n- Bob\n- Charlie\n\n## Updates\n...",
        "template": "standup",
        "keywords": ["standup", "daily", "team-a"]
    },
    {
        "title": "Sprint Planning - Jan 16", 
        "content": "## Sprint Goal\nComplete user authentication\n\n## Stories\n...",
        "keywords": ["sprint", "planning"]
    }
]
```

### Example: Import Templates

**templates.yaml**:
```yaml
- title: Bug Report Template
  content: |
    ## Bug Description
    
    ## Steps to Reproduce
    1. 
    2. 
    
    ## Expected Behavior
    
    ## Actual Behavior
    
    ## Environment
    - OS: 
    - Version:
  is_template: true

- title: Feature Request Template
  content: |
    ## Feature Description
    
    ## Use Case
    
    ## Proposed Solution
    
    ## Alternatives Considered
  is_template: true
```

### Example: Bulk Import from CSV

**notes_export.csv**:
```csv
title,content,keywords
"API Documentation","# REST API\n\n## Endpoints\n\n### GET /users\nReturns list of users...","api,docs"
"Database Schema","## Users Table\n- id: UUID\n- name: VARCHAR(255)\n- email: VARCHAR(255)","database,schema"
```

## Troubleshooting

### Common Issues

1. **Templates not appearing after import**
   - **Solution**: Restart the application after importing templates
   - Templates are loaded at startup from the config file

2. **Import fails with "Invalid format"**
   - **Solution**: Check that JSON/YAML files are properly formatted
   - Use a validator tool to check syntax

3. **Missing title or content**
   - **Solution**: Ensure all notes have both `title` and `content` fields
   - For plain text files, content cannot be empty

4. **Character encoding issues**
   - **Solution**: Save files in UTF-8 encoding
   - The importer expects UTF-8 encoded text files

5. **Large files timeout**
   - **Solution**: Split large imports into smaller files
   - Consider using CSV format for bulk imports

### Debug Mode

Enable debug logging to see detailed import information:
```bash
# Check logs for import details
tail -f ~/.local/share/tldw_cli/logs/app.log | grep -i "note.*import"
```

### File Size Limits

While there are no hard limits, consider:
- JSON/YAML files over 10MB may be slow to parse
- Each note's content should be under 1MB for optimal performance
- CSV files can handle thousands of notes efficiently

## Best Practices

1. **Organize Templates**: Group related templates in single files
2. **Use Metadata**: Include keywords and template associations
3. **Validate First**: Test with small files before bulk imports
4. **Backup**: Export existing notes before large imports
5. **Template Naming**: Use descriptive template names for easy selection

## Future Enhancements

Planned improvements:
- Drag-and-drop file import
- Direct template editing UI
- Import from cloud storage
- Export templates for sharing
- Template marketplace integration