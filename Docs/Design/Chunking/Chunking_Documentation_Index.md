# Chunking System Documentation Index

## Overview

The chunking system is a sophisticated text segmentation module that provides flexible, template-based document processing capabilities for the tldw_chatbook application. This index provides quick access to all chunking-related documentation.

## Documentation Structure

### 📋 [Chunking Architecture](Chunking_Architecture.md)
Comprehensive overview of the system architecture, design philosophy, and integration points.
- System components and data flow
- Design decisions and rationale
- Performance considerations
- Security and error handling

### 📚 [API Reference](Chunking_API_Reference.md)
Complete API documentation for all classes, methods, and functions.
- Core Chunker class API
- Template system API
- Configuration options
- Code examples

### 🎨 [Templates Guide](Chunking_Templates_Guide.md)
In-depth guide to the template system and creating custom chunking strategies.
- Template structure and syntax
- Built-in templates documentation
- Creating custom templates
- Operations reference

### 🔧 [Implementation Details](Chunking_Implementation_Details.md)
Technical deep-dive into algorithms and internal implementation.
- Algorithm specifications
- Language-specific handling
- Memory management
- Performance optimizations

### 💼 [Use Cases](Chunking_Use_Cases.md)
Practical guide with real-world scenarios and solutions.
- Common use cases with code examples
- Performance optimization strategies
- Troubleshooting guide
- Migration from legacy code

### 🚀 [Roadmap](Chunking_Roadmap.md)
Future enhancements and development plans.
- Short-term improvements
- Long-term vision
- Performance targets
- Community features

## Quick Links

### Getting Started
1. Read the [Architecture](Chunking_Architecture.md) overview
2. Review [Use Cases](Chunking_Use_Cases.md) for your scenario
3. Check the [API Reference](Chunking_API_Reference.md) for implementation

### Advanced Usage
1. Study the [Templates Guide](Chunking_Templates_Guide.md)
2. Understand [Implementation Details](Chunking_Implementation_Details.md)
3. Contribute to the [Roadmap](Chunking_Roadmap.md)

## Related Documentation

- [Main Architecture Document](../Architecture_and_Design.md) - Overall system architecture
- [RAG Search Modes](../../Development/rag_search_modes.md) - How chunking integrates with RAG
- [Migration Guide](../../../tldw_chatbook/Chunking/MIGRATION_GUIDE.md) - Migrating to template system
- [Template Examples](../../../tldw_chatbook/Chunking/templates/example_usage.py) - Working code examples

## Key Concepts

### Chunking Methods
- **Words**: Language-aware word-based chunking
- **Sentences**: Grammatical boundary preservation
- **Paragraphs**: Document structure maintenance
- **Tokens**: Precise LLM context management
- **Semantic**: Similarity-based grouping
- **Specialized**: JSON, XML, e-book chapters

### Template System
- JSON-based configuration
- Multi-stage pipelines
- Extensible operations
- Template inheritance

### Language Support
- Automatic language detection
- Chinese (jieba), Japanese (fugashi)
- Fallback strategies
- Mixed-language handling

## Quick Reference

### Basic Usage
```python
from tldw_chatbook.Chunking import Chunker

# Simple chunking
chunker = Chunker()
chunks = chunker.chunk_text(text, method="sentences")

# Template-based
chunker = Chunker(template="academic_paper")
chunks = chunker.chunk_text(text)
```

### Creating Custom Template
```json
{
  "name": "my_template",
  "pipeline": [
    {"stage": "preprocess", "operations": [...]},
    {"stage": "chunk", "method": "semantic"},
    {"stage": "postprocess", "operations": [...]}
  ]
}
```

## Support

For questions or issues:
1. Check the [Use Cases](Chunking_Use_Cases.md) troubleshooting section
2. Review [Implementation Details](Chunking_Implementation_Details.md) for technical issues
3. See the [Roadmap](Chunking_Roadmap.md) for planned features
4. Open an issue on the project repository