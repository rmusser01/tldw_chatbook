# Local File Ingestion

This module provides programmatic access to ingest local files into the tldw_chatbook Media database without using the UI.

## Features

- **Single File Ingestion**: Process individual files with full control over options
- **Batch Processing**: Ingest multiple files with common settings
- **Directory Scanning**: Automatically find and process all supported files in a directory
- **Full Processing Pipeline**: Leverages existing processors for all supported file types
- **Flexible Options**: Support for chunking, analysis, custom prompts, and more

## Supported File Types

### Documents
- PDF (`.pdf`)
- Microsoft Word (`.doc`, `.docx`)
- OpenDocument Text (`.odt`)
- Rich Text Format (`.rtf`)

### E-books
- EPUB (`.epub`)
- MOBI (`.mobi`)
- AZW (`.azw`, `.azw3`)
- FictionBook (`.fb2`)

### Web Pages
- HTML (`.html`, `.htm`)

### Text Files
- Plain Text (`.txt`)
- Markdown (`.md`, `.markdown`)
- reStructuredText (`.rst`)
- Log files (`.log`)
- CSV (`.csv`)

### Media
- Images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`)
- Audio (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.aac`, `.wma`, `.opus`)
- Video (`.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.mpg`, `.mpeg`)

## Quick Start

```python
from tldw_chatbook.Local_Ingestion import ingest_local_file
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

# Initialize database
media_db = MediaDatabase("/path/to/media_db.sqlite", client_id="my_script")

# Ingest a single file (raises FileIngestionError / FileNotFoundError on failure)
result = ingest_local_file(
    file_path="/path/to/document.pdf",
    media_db=media_db,
    keywords=["important", "document"]
)

print(f"Successfully ingested with media_id: {result['media_id']}")
```

## API Reference

### `ingest_local_file()`

Process and ingest a single local file.

**Parameters:**
- `file_path` (str/Path): Path to the file to ingest
- `media_db` (MediaDatabase): Database instance for storage
- `title` (str, optional): Override the document title
- `author` (str, optional): Set the document author
- `keywords` (list, optional): Keywords to associate with the content
- `custom_prompt` (str, optional): Custom analysis prompt
- `system_prompt` (str, optional): System prompt for analysis
- `perform_analysis` (bool): Whether to analyze/summarize (default: False)
- `api_name` (str, optional): API provider for analysis
- `api_key` (str, optional): API key if not using config default
- `chunk_options` (dict, optional): Chunking configuration. Omitting the
  argument means "chunk with defaults"; an explicit `None` disables
  chunking for every type; `{}` or a populated dict chunks with
  defaults/overrides. Keys:
  - `method`: Chunking method ('semantic', 'tokens', 'sentences', 'paragraphs', 'words', 'ebook_chapters')
  - `size` / `max_size`: Chunk size (default varies by method)
  - `overlap`: Overlap between chunks (default varies by method)
  - `adaptive`: Use adaptive chunking (bool)
  - `multi_level`: Use multi-level chunking (bool)
  - `language`: Language code for semantic chunking
- `metadata` (dict, optional): Additional metadata to store with the media

**Returns:**
Dictionary with:
- `media_id` (int): Database ID of ingested content
- `title` (str): Final title used
- `author` (str): Author used
- `content_length` (int): Length of extracted content
- `chunks_created` (int): Number of chunks created
- `keywords` (list): Keywords associated with the media
- `analysis`: Analysis results (if performed)
- `file_type` (str): Detected media type
- `file_path` (str): Path of processed file

**Raises:**
- `FileIngestionError`: If ingestion fails
- `FileNotFoundError`: If file doesn't exist

### `batch_ingest_files()`

Process multiple files with common options.

**Parameters:**
- `file_paths` (list): List of file paths to process
- `media_db` (MediaDatabase): Database instance
- `common_keywords` (list, optional): Keywords for all files
- `perform_analysis` (bool): Whether to analyze all files (default: False)
- `api_name` (str, optional): API provider for analysis
- `api_key` (str, optional): API key for analysis
- `chunk_options` (dict, optional): Chunking options for all files (same semantics as `ingest_local_file`)
- `stop_on_error` (bool): Stop on first error (default: False)

**Returns:**
List of result dictionaries for each file. Failed files get a
`{"file_path", "error", "success": False}` entry instead of a result
(when `stop_on_error` is False).

**Raises:**
- `FileIngestionError`: If `stop_on_error` is True and an error occurs

### `ingest_directory()`

Scan and ingest all supported files in a directory.

**Parameters:**
- `directory_path` (str/Path): Directory to scan
- `media_db` (MediaDatabase): Database instance
- `recursive` (bool): Include subdirectories (default: False)
- `file_types` (list, optional): Limit to specific media types (e.g. `['pdf', 'document']`); None processes all supported types
- `exclude_patterns` (list, optional): Filename patterns to exclude (e.g. `['*.tmp', 'draft_*']`)
- `**kwargs`: Additional arguments passed to `ingest_local_file`

**Returns:**
List of result dictionaries for each file found.

## Examples

### Advanced Ingestion with Analysis

```python
result = ingest_local_file(
    file_path="/path/to/research.pdf",
    media_db=media_db,
    title="2024 Research Findings",
    author="Dr. Smith",
    keywords=["research", "2024", "findings"],
    perform_analysis=True,
    api_name="openai",
    custom_prompt="Summarize the key findings and implications",
    chunk_options={
        "method": "semantic",
        "size": 1000,
        "overlap": 200
    }
)
```

### Batch Processing with Common Settings

```python
files = [
    "/docs/report1.pdf",
    "/docs/report2.docx",
    "/docs/notes.txt"
]

results = batch_ingest_files(
    file_paths=files,
    media_db=media_db,
    common_keywords=["2024", "quarterly"],
    chunk_options={"method": "tokens", "size": 500}
)

# Check results (failed files carry success=False and an error message)
for r in results:
    if r.get("success") is False:
        print(f"{r['file_path']}: ✗ {r['error']}")
    else:
        print(f"{r['file_path']}: ✓ media_id {r['media_id']}")
```

### Directory Ingestion

```python
# Ingest all PDFs and e-books in a directory tree
results = ingest_directory(
    directory_path="/home/user/Documents",
    media_db=media_db,
    recursive=True,
    file_types=['pdf', 'ebook'],
    keywords=["archive", "2024"],
    perform_analysis=False  # Just store, don't analyze
)

print(f"Processed {len(results)} files")
```

### Integration with Existing Code

```python
# In a larger application
class DocumentProcessor:
    def __init__(self, db_path: str):
        self.media_db = MediaDatabase(db_path, client_id="doc_processor")
    
    def process_upload(self, file_path: str, user_tags: list):
        """Process a user-uploaded document."""
        try:
            result = ingest_local_file(
                file_path=file_path,
                media_db=self.media_db,
                keywords=user_tags,
                perform_analysis=True,
                api_name=self.config.get('analysis_api')
            )
        except FileIngestionError as e:
            self.handle_error(str(e))
            return None
        
        self.notify_user(f"Document processed: {result['title']}")
        return result['media_id']
```

## Error Handling

`ingest_local_file()` reports failure by raising:

- `FileNotFoundError`: the file doesn't exist
- `FileIngestionError`: unsupported file type, processing failure, or database storage failure (the message names the failing stage)

```python
from tldw_chatbook.Local_Ingestion import FileIngestionError

try:
    result = ingest_local_file(file_path="/path/to/file.pdf", media_db=db)
except FileNotFoundError:
    print("File doesn't exist")
except FileIngestionError as e:
    print(f"Ingestion failed: {e}")
```

`batch_ingest_files()` catches per-file errors (unless `stop_on_error=True`)
and appends a `{"file_path", "error", "success": False}` entry to the
results for each failed file.

## Performance Considerations

1. **Large Files**: The processors handle large files by chunking. Memory usage is managed automatically.

2. **Batch Processing**: When processing many files, consider:
   - Using `batch_ingest_files()` for better progress tracking
   - Setting `stop_on_error=True` if you want to halt on failures
   - Processing in smaller batches for very large collections

3. **Analysis**: LLM analysis can be slow and costly. Consider:
   - Skipping analysis during initial ingestion
   - Running analysis as a separate step
   - Using cheaper/faster models for large batches

## Database Considerations

- The pipeline writes through `persist_parsed_media()`, whose single
  `add_media_with_keywords()` call is the only database write in the ingest path
- Content is deduplicated by URL and content hash
- Use `persist_parsed_media(payload, media_db, overwrite_existing=True)` to update existing content — this governs **live** rows only.
  A match that is sitting in Trash (`is_trash = 1`) is never modified by
  `overwrite_existing`; restoring it requires the explicit
  `restore_trashed=True` opt-in at the `add_media_with_keywords` level
  (task-4026; the Library ingest writer `persist_parsed_media` passes it)
- All operations are transactional

## See Also

- Individual processor documentation in their respective modules
- Media database documentation in `DB/Client_Media_DB_v2.py`