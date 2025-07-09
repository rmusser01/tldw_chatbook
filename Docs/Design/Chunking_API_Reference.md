# Chunking System API Reference

## Table of Contents
1. [Core Classes](#core-classes)
2. [Functions](#functions)
3. [Template System API](#template-system-api)
4. [Language Support API](#language-support-api)
5. [Token Support API](#token-support-api)
6. [Configuration Options](#configuration-options)
7. [Exceptions](#exceptions)

## Core Classes

### Chunker

The main class for performing text chunking operations.

```python
class Chunker:
    def __init__(self,
                 options: Optional[Dict[str, Any]] = None,
                 tokenizer_name_or_path: str = "gpt2",
                 template: Optional[str] = None,
                 template_manager: Optional[ChunkingTemplateManager] = None)
```

#### Parameters
- **options** (`Optional[Dict[str, Any]]`): Custom chunking options to override defaults
- **tokenizer_name_or_path** (`str`): HuggingFace tokenizer name or path. Default: "gpt2"
- **template** (`Optional[str]`): Name of chunking template to use
- **template_manager** (`Optional[ChunkingTemplateManager]`): Custom template manager instance

#### Methods

##### chunk_text()
```python
def chunk_text(self,
               text: str,
               method: Optional[str] = None,
               llm_call_function: Optional[Callable] = None,
               llm_api_config: Optional[Dict[str, Any]] = None,
               use_template: Optional[bool] = None) -> List[Union[str, Dict[str, Any]]]
```

Main method to chunk text based on the specified method or template.

**Parameters:**
- **text** (`str`): The text to chunk
- **method** (`Optional[str]`): Override the chunking method
- **llm_call_function** (`Optional[Callable]`): LLM function for rolling_summarize
- **llm_api_config** (`Optional[Dict]`): LLM configuration
- **use_template** (`Optional[bool]`): Force use/bypass of template

**Returns:** List of chunks (strings or dictionaries for structured methods)

**Example:**
```python
chunker = Chunker(template="academic_paper")
chunks = chunker.chunk_text(text)
```

##### detect_language()
```python
def detect_language(self, text: str) -> str
```

Detects the language of the given text.

**Parameters:**
- **text** (`str`): Text to analyze

**Returns:** Language code (e.g., 'en', 'zh-cn'). Defaults to 'en' if detection fails.

**Example:**
```python
language = chunker.detect_language("这是中文文本")
# Returns: 'zh-cn'
```

## Functions

### improved_chunking_process()
```python
def improved_chunking_process(
    text: str,
    chunk_options_dict: Optional[Dict[str, Any]] = None,
    tokenizer_name_or_path: str = "gpt2",
    template: Optional[str] = None,
    template_manager: Optional[ChunkingTemplateManager] = None,
    llm_call_function_for_chunker: Optional[Callable] = None,
    llm_api_config_for_chunker: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

High-level function that performs chunking with metadata enrichment.

**Returns:** List of dictionaries with 'text' and 'metadata' keys

**Example:**
```python
chunks = improved_chunking_process(
    text,
    template="semantic",
    chunk_options_dict={"max_size": 1000}
)
```

### chunk_for_embedding()
```python
def chunk_for_embedding(
    text: str,
    file_name: str,
    custom_chunk_options: Optional[Dict[str, Any]] = None,
    tokenizer_name_or_path: str = "gpt2",
    llm_call_function: Optional[Callable] = None,
    llm_api_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

Prepares chunks specifically for embedding generation with context headers.

**Returns:** List of dictionaries with embedding-specific formatting

**Example:**
```python
embedding_chunks = chunk_for_embedding(
    text,
    "research_paper.pdf",
    custom_chunk_options={"method": "semantic"}
)
```

### process_document_with_metadata()
```python
def process_document_with_metadata(
    text: str,
    chunk_options_dict: Dict[str, Any],
    document_metadata: Dict[str, Any],
    tokenizer_name_or_path: str = "gpt2",
    llm_call_function: Optional[Callable] = None,
    llm_api_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Processes a document and associates document-level metadata with chunks.

**Example:**
```python
result = process_document_with_metadata(
    text,
    {"method": "paragraphs"},
    {"author": "John Doe", "date": "2024-01-01"}
)
```

## Template System API

### ChunkingTemplate
```python
class ChunkingTemplate(BaseModel):
    name: str
    description: Optional[str] = None
    base_method: str = "words"
    pipeline: List[ChunkingStage]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_template: Optional[str] = None
```

Defines a complete chunking strategy template.

### ChunkingTemplateManager
```python
class ChunkingTemplateManager:
    def __init__(self,
                 templates_dir: Optional[Path] = None,
                 user_templates_dir: Optional[Path] = None)
```

Manages loading, caching, and validation of chunking templates.

#### Key Methods

##### load_template()
```python
def load_template(self, template_name: str) -> Optional[ChunkingTemplate]
```

Load a template by name, checking cache first.

##### save_template()
```python
def save_template(self, template: ChunkingTemplate, user_template: bool = True)
```

Save a template to disk.

##### register_operation()
```python
def register_operation(self, name: str, operation: Callable)
```

Register a custom operation for use in templates.

**Example:**
```python
def custom_op(text: str, chunks: List[str], options: Dict) -> List[str]:
    # Custom logic
    return modified_chunks

manager.register_operation("my_custom_op", custom_op)
```

##### get_available_templates()
```python
def get_available_templates(self) -> List[str]
```

Get list of all available template names.

### ChunkingPipeline
```python
class ChunkingPipeline:
    def __init__(self, template_manager: ChunkingTemplateManager)
    
    def execute(self,
                text: str,
                template: ChunkingTemplate,
                chunker_instance: Chunker,
                **kwargs) -> List[Dict[str, Any]]
```

Executes chunking templates with multi-stage processing.

## Language Support API

### LanguageChunkerFactory
```python
class LanguageChunkerFactory:
    @staticmethod
    def get_chunker(language: str) -> LanguageChunker
    
    @staticmethod
    def get_available_languages() -> List[str]
```

Factory for creating language-specific chunkers.

### Language Chunker Protocol
```python
class LanguageChunker(Protocol):
    def tokenize_words(self, text: str) -> List[str]
    def tokenize_sentences(self, text: str) -> List[str]
```

## Token Support API

### TokenBasedChunker
```python
class TokenBasedChunker:
    def __init__(self, tokenizer_name_or_path: str = "gpt2")
    
    def chunk_by_tokens(self,
                        text: str,
                        max_tokens: int,
                        overlap_tokens: int = 0) -> List[str]
    
    def count_tokens(self, text: str) -> int
    
    def is_transformers_available(self) -> bool
```

Token-based text chunker with optional transformers support.

## Configuration Options

### Chunking Options Dictionary

All chunking methods accept these common options:

```python
{
    # Core options
    'method': str,              # Chunking method name
    'max_size': int,            # Maximum chunk size
    'overlap': int,             # Overlap between chunks
    'language': str,            # Language code (auto-detected if None)
    
    # Advanced options
    'adaptive': bool,           # Enable adaptive chunking
    'multi_level': bool,        # Enable multi-level chunking
    
    # Method-specific options
    'custom_chapter_pattern': str,     # For ebook_chapters
    'semantic_similarity_threshold': float,  # For semantic
    'semantic_overlap_sentences': int,       # For semantic
    'json_chunkable_data_key': str,         # For json
    
    # Template options
    'template': str,            # Template name to use
}
```

### Available Chunking Methods

1. **words** - Word-based chunking
   - `max_size`: Maximum words per chunk
   - `overlap`: Word overlap between chunks

2. **sentences** - Sentence-based chunking
   - `max_size`: Maximum sentences per chunk
   - `overlap`: Sentence overlap

3. **paragraphs** - Paragraph-based chunking
   - `max_size`: Maximum paragraphs per chunk
   - `overlap`: Paragraph overlap

4. **tokens** - Token-based chunking
   - `max_size`: Maximum tokens per chunk
   - `overlap`: Token overlap

5. **semantic** - Semantic similarity-based chunking
   - `max_size`: Maximum chunk size (words/tokens)
   - `semantic_similarity_threshold`: Similarity threshold (0-1)
   - `unit`: Size unit ('words', 'tokens', 'characters')

6. **json** - JSON-aware chunking
   - `max_size`: Items per chunk
   - `overlap`: Item overlap
   - `json_chunkable_data_key`: Key containing data to chunk

7. **xml** - XML-aware chunking
   - `max_size`: Maximum words per chunk
   - `overlap`: Element overlap

8. **ebook_chapters** - Chapter-based chunking
   - `custom_chapter_pattern`: Custom regex for chapters
   - `max_size`: Sub-chunk large chapters if > 0

9. **rolling_summarize** - Progressive summarization
   - `detail`: Summarization detail level (0-1)
   - `min_chunk_tokens`: Minimum tokens per chunk
   - `chunk_delimiter`: Delimiter for splitting

## Exceptions

### Base Exception
```python
class ChunkingError(Exception):
    """Base exception for chunking errors."""
```

### Specific Exceptions
```python
class InvalidChunkingMethodError(ChunkingError):
    """Raised when an invalid chunking method is specified."""

class InvalidInputError(ChunkingError):
    """Raised for invalid input data."""

class LanguageDetectionError(ChunkingError):
    """Raised when language detection fails critically."""
```

## Usage Examples

### Basic Usage
```python
from tldw_chatbook.Chunking import Chunker

# Simple word-based chunking
chunker = Chunker()
chunks = chunker.chunk_text("Your text here", method="words")

# Using options
chunker = Chunker(options={
    'method': 'sentences',
    'max_size': 5,
    'overlap': 1
})
chunks = chunker.chunk_text(text)
```

### Template Usage
```python
# Use built-in template
chunker = Chunker(template="academic_paper")
chunks = chunker.chunk_text(academic_text)

# Override template options
chunker = Chunker(
    template="academic_paper",
    options={'max_size': 1200}
)
```

### Advanced Usage
```python
# Custom template with pipeline
from tldw_chatbook.Chunking import (
    ChunkingTemplate, ChunkingStage, ChunkingOperation
)

template = ChunkingTemplate(
    name="custom",
    pipeline=[
        ChunkingStage(
            stage="preprocess",
            operations=[
                ChunkingOperation(type="normalize_whitespace")
            ]
        ),
        ChunkingStage(
            stage="chunk",
            method="semantic",
            options={"max_size": 1000}
        ),
        ChunkingStage(
            stage="postprocess",
            operations=[
                ChunkingOperation(
                    type="add_context",
                    params={"context_size": 2}
                )
            ]
        )
    ]
)
```

### Error Handling
```python
try:
    chunks = chunker.chunk_text(text, method="invalid_method")
except InvalidChunkingMethodError as e:
    print(f"Invalid method: {e}")
    # Fallback to default
    chunks = chunker.chunk_text(text)
```

## Performance Tips

1. **Reuse Chunker instances** - Tokenizers are cached
2. **Batch processing** - Process multiple documents with same settings
3. **Choose appropriate methods** - Token-based for LLMs, semantic for search
4. **Configure overlap carefully** - Balance context vs. redundancy
5. **Use templates** - Avoid recreating complex configurations

## Thread Safety

The Chunker class is **not thread-safe**. For concurrent processing:
- Create separate Chunker instances per thread
- Or use thread-local storage
- Or implement external synchronization

## Version Compatibility

- Python 3.8+ required
- Optional dependencies versioning:
  - transformers: 4.0+
  - jieba: any recent version
  - fugashi: 1.0+
  - nltk: 3.5+