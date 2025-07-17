# Chunking Template System Migration Guide

This guide helps you migrate from the traditional chunking approach to the new template-based system.

## What's New

The chunking module now supports:
- **Template-based configuration** - Define chunking strategies in JSON files
- **Multi-stage pipelines** - Preprocessing, chunking, and postprocessing stages
- **Extensible operations** - Add custom operations for specialized needs
- **Template inheritance** - Build on existing templates
- **Domain-specific templates** - Pre-built templates for common use cases

## Backward Compatibility

**All existing code continues to work without changes.** The template system is optional and additive.

## Migration Examples

### Before (Traditional Approach)
```python
from tldw_chatbook.Chunking.Chunk_Lib import Chunker

# Basic chunking
chunker = Chunker(options={
    'method': 'words',
    'max_size': 500,
    'overlap': 100
})
chunks = chunker.chunk_text(text)
```

### After (Template Approach)
```python
from tldw_chatbook.Chunking import Chunker

# Use a template
chunker = Chunker(template="words")
chunks = chunker.chunk_text(text)

# Or override template options
chunker = Chunker(
    template="words",
    options={'max_size': 500, 'overlap': 100}
)
chunks = chunker.chunk_text(text)
```

## Benefits of Templates

### 1. Reusability
Define once, use everywhere:
```python
# In config.toml
[chunking_config]
template = "academic_paper"

# In code - automatically uses the configured template
chunker = Chunker()
```

### 2. Complex Pipelines
Templates support multi-stage processing:
```json
{
  "pipeline": [
    {"stage": "preprocess", "operations": [...]},
    {"stage": "chunk", "method": "semantic"},
    {"stage": "postprocess", "operations": [...]}
  ]
}
```

### 3. Domain Optimization
Use specialized templates:
```python
# For code documentation
chunker = Chunker(template="code_documentation")

# For legal documents
chunker = Chunker(template="legal_document")

# For conversations
chunker = Chunker(template="conversation")
```

## Creating Custom Templates

### Option 1: JSON File
Create `~/.config/tldw_cli/chunking_templates/my_template.json`:
```json
{
  "name": "my_template",
  "base_method": "sentences",
  "pipeline": [
    {
      "stage": "chunk",
      "method": "sentences",
      "options": {"max_size": 5}
    }
  ]
}
```

### Option 2: Programmatically
```python
from tldw_chatbook.Chunking import ChunkingTemplate, ChunkingTemplateManager

template = ChunkingTemplate(
    name="my_template",
    base_method="sentences",
    pipeline=[...]
)

manager = ChunkingTemplateManager()
manager.save_template(template)
```

## Common Patterns

### Adding Preprocessing
```json
{
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {"type": "normalize_whitespace"},
        {"type": "extract_metadata", "params": {"patterns": ["title", "author"]}}
      ]
    },
    {"stage": "chunk", "method": "semantic"}
  ]
}
```

### Adding Context
```json
{
  "pipeline": [
    {"stage": "chunk", "method": "words"},
    {
      "stage": "postprocess",
      "operations": [
        {"type": "add_context", "params": {"context_size": 2}}
      ]
    }
  ]
}
```

## Gradual Migration

You don't need to migrate everything at once:

1. **Start with built-in templates** - Try existing templates that match your use case
2. **Customize as needed** - Override specific options while using templates
3. **Create custom templates** - For repeated patterns in your codebase
4. **Share templates** - Export templates for team use

## Performance Notes

- Templates add minimal overhead (template loading is cached)
- Pipeline operations are optimized
- Same chunking performance as before

## Getting Help

- See `templates/README.md` for template documentation
- Run `templates/example_usage.py` for working examples
- Check existing templates in `templates/` directory
- Use `ChunkingTemplateManager.get_available_templates()` to list templates

## FAQ

**Q: Do I have to use templates?**
A: No, all existing code works without modification.

**Q: Can I mix approaches?**
A: Yes, you can use templates with option overrides.

**Q: Are templates slower?**
A: No, template loading is cached and adds negligible overhead.

**Q: Can I convert existing options to a template?**
A: Yes, create a template with your current options in the chunk stage.

**Q: What about custom chunking methods?**
A: Custom methods still work. Templates organize how methods are applied.