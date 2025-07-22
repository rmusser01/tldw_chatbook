# Chunking Interop Library Documentation

## Overview

The `ChunkingInteropService` provides a centralized, thread-safe interface for managing chunking templates and per-document configurations. It follows the same pattern as the `NotesInteropService`, providing a clean abstraction over the database operations.

**Location**: `/tldw_chatbook/Chunking/chunking_interop_library.py`

## Key Features

### 1. Template Management
- Full CRUD operations for chunking templates
- System template protection (cannot modify/delete)
- Template caching for performance
- Import/Export functionality
- Template validation

### 2. Document Configuration
- Per-document chunking configuration storage
- Configuration retrieval with JSON parsing
- Clear/reset functionality
- Find documents using specific templates

### 3. Thread Safety
- Thread-safe template cache with locking
- Safe concurrent access to database operations

### 4. Error Handling
- Custom exceptions for specific error cases:
  - `ChunkingTemplateError` - Base exception
  - `TemplateNotFoundError` - Template not found
  - `SystemTemplateError` - Attempting to modify system template
  - `InputError` - Validation failures

## Usage Examples

### Initialize the Service

```python
from tldw_chatbook.Chunking.chunking_interop_library import get_chunking_service
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

# Get media database instance
media_db = MediaDatabase(db_path, client_id="app")

# Create service instance
chunking_service = get_chunking_service(media_db)
```

### Template Operations

```python
# Get all templates
templates = chunking_service.get_all_templates(include_system=True)

# Get specific template
template = chunking_service.get_template_by_id(1)
template = chunking_service.get_template_by_name("academic_paper")

# Create new template
template_id = chunking_service.create_template(
    name="My Custom Template",
    description="Template for research papers",
    template_json={
        "name": "custom",
        "base_method": "structural",
        "pipeline": [
            {
                "stage": "chunk",
                "method": "structural",
                "options": {"max_size": 500, "overlap": 50}
            }
        ]
    }
)

# Update template
chunking_service.update_template(
    template_id=5,
    name="Updated Name",
    description="Updated description"
)

# Delete template (only custom templates)
chunking_service.delete_template(template_id=5)

# Duplicate template
new_id = chunking_service.duplicate_template(
    template_id=1,
    new_name="Academic Paper v2"
)
```

### Document Configuration

```python
# Set configuration for a document
config = {
    "template": "academic_paper",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "method": "structural",
    "enable_late_chunking": True
}
chunking_service.set_document_config(media_id=123, config=config)

# Get configuration
config = chunking_service.get_document_config(media_id=123)

# Clear configuration (revert to defaults)
chunking_service.clear_document_config(media_id=123)

# Find all documents using a template
docs = chunking_service.get_documents_using_template("academic_paper")
```

### Import/Export

```python
# Export template
export_data = chunking_service.export_template(template_id=1)
# Save to file
with open("template.json", "w") as f:
    json.dump(export_data, f, indent=2)

# Import template
with open("template.json", "r") as f:
    import_data = json.load(f)
    
new_template_id = chunking_service.import_template(
    import_data,
    name_suffix=" (Imported)"  # Added if name conflicts
)
```

### Statistics

```python
stats = chunking_service.get_template_statistics()
# Returns:
# {
#     'total_templates': 10,
#     'system_templates': 5,
#     'custom_templates': 5,
#     'configured_documents': 42,
#     'most_used_templates': [
#         {'template': 'academic_paper', 'count': 15},
#         {'template': 'general', 'count': 12}
#     ]
# }
```

### Template Validation

```python
# Validate template JSON structure
is_valid, error_msg = chunking_service.validate_template_json({
    "name": "test",
    "base_method": "words",
    "pipeline": [{"stage": "chunk", "method": "words"}]
})

if not is_valid:
    print(f"Invalid template: {error_msg}")
```

## Widget Integration

The service has been integrated into all relevant widgets:

### ChunkingTemplatesWidget
- Uses service for all template CRUD operations
- Leverages caching for performance
- Proper error handling with user notifications

### ChunkingTemplateEditor
- Creates/updates templates through the service
- Validates templates before saving
- Handles InputError for user-friendly messages

### MediaDetailsWidget
- Gets/sets document configurations via service
- Loads template configurations
- Clear functionality for resetting

## Benefits

1. **Centralized Logic**: All chunking database operations in one place
2. **Consistency**: Uniform error handling and validation
3. **Performance**: Built-in caching reduces database queries
4. **Thread Safety**: Safe for concurrent access
5. **Maintainability**: Easy to extend and modify
6. **Testing**: Can mock the service for unit tests

## Error Handling

The service provides specific exceptions for different error cases:

```python
try:
    chunking_service.delete_template(1)
except SystemTemplateError:
    # Cannot delete system templates
    notify("Cannot delete system templates")
except TemplateNotFoundError:
    # Template doesn't exist
    notify("Template not found")
except ChunkingTemplateError as e:
    # Other template errors
    notify(f"Error: {str(e)}")
```

## Future Enhancements

1. **Bulk Operations**: Add methods for bulk template operations
2. **Template Versioning**: Track template version history
3. **Usage Analytics**: More detailed usage statistics
4. **Template Validation Rules**: Customizable validation rules
5. **Async Support**: Add async methods for better performance