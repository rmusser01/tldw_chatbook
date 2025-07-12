# Chunking Templates Guide

## Introduction

The chunking template system provides a declarative way to define complex text chunking strategies. Instead of writing code for each chunking scenario, you can define reusable templates that describe multi-stage processing pipelines with preprocessing, chunking, and postprocessing operations.

## Template Structure

### Basic Template Anatomy

```json
{
  "name": "template_name",
  "description": "What this template does",
  "base_method": "default_method_if_no_pipeline",
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [...]
    },
    {
      "stage": "chunk",
      "method": "chunking_method",
      "options": {...}
    },
    {
      "stage": "postprocess",
      "operations": [...]
    }
  ],
  "metadata": {
    "suitable_for": ["use_case1", "use_case2"],
    "custom_key": "custom_value"
  },
  "parent_template": "optional_parent_name"
}
```

### Field Descriptions

#### Top-Level Fields

- **name** (required): Unique identifier for the template
- **description**: Human-readable description
- **base_method**: Default chunking method if pipeline doesn't specify
- **pipeline**: Array of processing stages
- **metadata**: Additional information about the template
- **parent_template**: Name of template to inherit from

#### Pipeline Stages

Templates support three types of stages executed in order:

1. **preprocess**: Prepare text before chunking
2. **chunk**: The actual chunking operation
3. **postprocess**: Transform chunks after creation

#### Stage Structure

```json
{
  "stage": "stage_type",
  "operations": [
    {
      "type": "operation_name",
      "params": {
        "param1": "value1"
      },
      "condition": "optional_condition_expression"
    }
  ],
  "method": "for_chunk_stage_only",
  "options": {
    "option1": "value1"
  }
}
```

## Built-in Templates

### Basic Templates

#### words.json
```json
{
  "name": "words",
  "description": "Basic word-based chunking with configurable size and overlap",
  "base_method": "words",
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {"type": "normalize_whitespace"}
      ]
    },
    {
      "stage": "chunk",
      "method": "words",
      "options": {
        "max_size": 400,
        "overlap": 200
      }
    },
    {
      "stage": "postprocess",
      "operations": [
        {
          "type": "filter_empty",
          "params": {"min_length": 10}
        }
      ]
    }
  ]
}
```

#### sentences.json
Preserves grammatical boundaries:
- Max sentences per chunk: 5
- Overlap: 1 sentence
- Filters very short chunks

#### paragraphs.json
Maintains document structure:
- Max paragraphs: 3
- Merges small chunks
- Preserves paragraph breaks

#### tokens.json
Precise LLM context management:
- Max tokens: 500
- Token overlap: 50
- Requires tokenizer

### Specialized Templates

#### semantic.json
Groups semantically related content:
- Uses TF-IDF similarity
- Configurable similarity threshold
- Adds context between chunks

#### ebook_chapters.json
Preserves book structure:
- Detects chapter boundaries
- Sub-chunks large chapters
- Multiple heading patterns

#### json.json
Handles structured JSON data:
- Preserves JSON structure
- Configurable data key
- Maintains metadata

#### xml.json
XML-aware chunking:
- Preserves element paths
- Includes attributes
- Configurable overlap

### Domain-Specific Templates

#### academic_paper.json
Optimized for research papers:
```json
{
  "name": "academic_paper",
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {
          "type": "extract_metadata",
          "params": {
            "patterns": ["abstract", "keywords"]
          }
        },
        {
          "type": "section_detection",
          "params": {
            "headers": [
              "Abstract", "Introduction", "Methods",
              "Results", "Discussion", "Conclusion"
            ]
          }
        }
      ]
    },
    {
      "stage": "chunk",
      "method": "semantic",
      "options": {
        "max_size": 800,
        "semantic_similarity_threshold": 0.7
      }
    }
  ]
}
```

#### code_documentation.json
For technical documentation:
- Detects code blocks
- Preserves code structure
- Merges small sections

#### legal_document.json
Legal text processing:
- Preserves clause structure
- Maintains legal hierarchy
- Adds context for references

#### conversation.json
Dialogue and transcripts:
- Preserves speaker turns
- Maintains conversation flow
- Overlaps for context

## Built-in Operations

### Preprocessing Operations

#### normalize_whitespace
Normalizes spaces and newlines:
```json
{
  "type": "normalize_whitespace",
  "params": {}
}
```

#### extract_metadata
Extracts metadata by patterns:
```json
{
  "type": "extract_metadata",
  "params": {
    "patterns": ["abstract", "keywords", "title", "author"]
  }
}
```
Supported patterns:
- `abstract`: Extracts abstract section
- `keywords`: Extracts comma-separated keywords
- Custom patterns can be added

#### section_detection
Detects document sections:
```json
{
  "type": "section_detection",
  "params": {
    "headers": ["Introduction", "Methods", "Results"]
  }
}
```

#### code_block_detection
Identifies code blocks:
```json
{
  "type": "code_block_detection",
  "params": {}
}
```
Detects:
- Fenced code blocks (```)
- Language specifications
- Block positions

### Postprocessing Operations

#### add_context
Adds surrounding chunk context:
```json
{
  "type": "add_context",
  "params": {
    "context_size": 2  // Number of surrounding chunks
  }
}
```

#### add_overlap
Adds text overlap between chunks:
```json
{
  "type": "add_overlap",
  "params": {
    "overlap_size": 50  // Characters to overlap
  }
}
```

#### filter_empty
Removes empty/short chunks:
```json
{
  "type": "filter_empty",
  "params": {
    "min_length": 10  // Minimum characters
  }
}
```

#### merge_small
Combines small chunks:
```json
{
  "type": "merge_small",
  "params": {
    "min_size": 200  // Minimum chunk size
  }
}
```

## Creating Custom Templates

### Step 1: Analyze Requirements

Consider:
- Content type and structure
- Downstream use (embeddings, LLM, search)
- Performance requirements
- Special handling needs

### Step 2: Design Pipeline

```json
{
  "name": "my_custom_template",
  "description": "Template for my specific use case",
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        // Add preprocessing operations
      ]
    },
    {
      "stage": "chunk",
      "method": "choose_method",
      "options": {
        // Configure chunking
      }
    },
    {
      "stage": "postprocess",
      "operations": [
        // Add postprocessing
      ]
    }
  ]
}
```

### Step 3: Save Template

Save to `~/.config/tldw_cli/chunking_templates/my_template.json`

### Step 4: Test Template

```python
from tldw_chatbook.Chunking import Chunker

chunker = Chunker(template="my_custom_template")
chunks = chunker.chunk_text(sample_text)
```

## Template Inheritance

Templates can inherit from other templates:

```json
{
  "name": "research_paper_large",
  "parent_template": "academic_paper",
  "pipeline": [
    {
      "stage": "chunk",
      "options": {
        "max_size": 1500  // Override parent's max_size
      }
    }
  ]
}
```

Inheritance rules:
- Child overrides parent settings
- Pipeline stages are merged
- Metadata is inherited
- Operations can be replaced

## Custom Operations

### Creating Custom Operations

```python
def my_custom_operation(text: str, chunks: List[str], options: Dict[str, Any]) -> Any:
    """
    Custom operation implementation.
    
    Args:
        text: Original text
        chunks: Current chunks (may be empty in preprocess)
        options: Operation parameters
        
    Returns:
        Modified chunks, extracted data, or modified text
    """
    # Implementation
    return result
```

### Registering Operations

```python
from tldw_chatbook.Chunking import ChunkingTemplateManager

manager = ChunkingTemplateManager()
manager.register_operation("my_custom_op", my_custom_operation)
```

### Using in Templates

```json
{
  "operations": [
    {
      "type": "my_custom_op",
      "params": {
        "option1": "value1"
      }
    }
  ]
}
```

## Conditional Operations

Operations can have conditions:

```json
{
  "type": "add_context",
  "params": {"context_size": 2},
  "condition": "text_length > 5000"
}
```

Available variables in conditions:
- `text_length`: Length of original text
- `chunk_count`: Number of chunks (postprocess only)
- `template`: Template name
- Custom variables from operations

## Template Metadata

Use metadata for:
- Documentation
- Tool integration
- Selection logic

```json
{
  "metadata": {
    "suitable_for": ["research", "academic", "scientific"],
    "requires": ["sklearn", "nltk"],
    "performance": "medium",
    "quality": "high",
    "custom_field": "custom_value"
  }
}
```

## Best Practices

### 1. Template Naming
- Use descriptive names
- Include domain/purpose
- Avoid generic names

### 2. Pipeline Design
- Keep stages focused
- Order operations logically
- Test each stage independently

### 3. Performance
- Preprocess expensive operations
- Cache operation results
- Profile with real data

### 4. Documentation
- Always include description
- Document custom operations
- Provide usage examples

### 5. Testing
```python
# Test template thoroughly
def test_template(template_name, test_texts):
    chunker = Chunker(template=template_name)
    for text in test_texts:
        chunks = chunker.chunk_text(text)
        # Validate results
        assert len(chunks) > 0
        assert all(len(chunk) > 0 for chunk in chunks)
```

## Common Patterns

### Pattern 1: Extract Then Chunk
```json
{
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {"type": "extract_metadata"},
        {"type": "section_detection"}
      ]
    },
    {
      "stage": "chunk",
      "method": "semantic"
    }
  ]
}
```

### Pattern 2: Clean and Normalize
```json
{
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {"type": "normalize_whitespace"},
        {"type": "remove_headers"}
      ]
    },
    {
      "stage": "chunk",
      "method": "paragraphs"
    }
  ]
}
```

### Pattern 3: Preserve Structure
```json
{
  "pipeline": [
    {
      "stage": "chunk",
      "method": "semantic"
    },
    {
      "stage": "postprocess",
      "operations": [
        {"type": "add_context"},
        {"type": "preserve_boundaries"}
      ]
    }
  ]
}
```

## Debugging Templates

### Enable Debug Logging
```python
import logging
logging.getLogger("tldw_chatbook.Chunking").setLevel(logging.DEBUG)
```

### Inspect Pipeline Execution
```python
from tldw_chatbook.Chunking import ChunkingPipeline, ChunkingTemplateManager

manager = ChunkingTemplateManager()
template = manager.load_template("my_template")
pipeline = ChunkingPipeline(manager)

# Add debugging
pipeline.context['debug'] = True
results = pipeline.execute(text, template, chunker)
```

### Common Issues

1. **Operations not executing**: Check conditions
2. **Empty chunks**: Verify filter operations
3. **Large chunks**: Check max_size in options
4. **Missing metadata**: Ensure operations return data

## Template Examples

### Email Thread Template
```json
{
  "name": "email_thread",
  "description": "Chunks email threads preserving conversation flow",
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {
          "type": "detect_email_boundaries",
          "params": {"patterns": ["From:", "Date:", "Subject:"]}
        }
      ]
    },
    {
      "stage": "chunk",
      "method": "custom_email",
      "options": {
        "preserve_threads": true,
        "max_emails_per_chunk": 5
      }
    }
  ]
}
```

### Social Media Template
```json
{
  "name": "social_media",
  "description": "Chunks social media posts preserving context",
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {"type": "detect_mentions"},
        {"type": "extract_hashtags"}
      ]
    },
    {
      "stage": "chunk",
      "method": "sentences",
      "options": {
        "max_size": 3,
        "preserve_posts": true
      }
    }
  ]
}
```

## Conclusion

The template system provides a powerful, flexible way to define chunking strategies without writing code. By combining built-in operations with custom logic, you can create sophisticated text processing pipelines tailored to your specific needs. Start with built-in templates, customize as needed, and share your templates with the community.