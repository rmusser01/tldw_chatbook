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
- Microsoft Word (`.docx`, `.doc`)
- OpenDocument Text (`.odt`)
- Rich Text Format (`.rtf`)
- PowerPoint (`.pptx`, `.ppt`)
- Excel (`.xlsx`, `.xls`)
- OpenDocument Spreadsheet (`.ods`)
- OpenDocument Presentation (`.odp`)

### E-books
- EPUB (`.epub`)
- MOBI (`.mobi`)
- AZW (`.azw`, `.azw3`)
- FictionBook (`.fb2`)

### Text Files
- Plain Text (`.txt`, `.text`)
- Markdown (`.md`, `.markdown`)
- reStructuredText (`.rst`)

### Structured Data
- XML (`.xml`)
- OPML (`.opml`)

## Quick Start

```python
from tldw_chatbook.Local_Ingestion import ingest_local_file
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

# Initialize database
media_db = MediaDatabase("/path/to/media_db.sqlite", client_id="my_script")

# Ingest a single file
result = ingest_local_file(
    file_path="/path/to/document.pdf",
    media_db=media_db,
    keywords=["important", "document"]
)

if result['success']:
    print(f"Successfully ingested with media_id: {result['media_id']}")
else:
    print(f"Failed: {result['message']}")
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
- `perform_chunking` (bool): Whether to chunk the content (default: True)
- `chunk_options` (dict, optional): Chunking configuration
  - `method`: Chunking method ('semantic', 'tokens', 'sentences', etc.)
  - `chunk_size`: Size of each chunk
  - `overlap`: Overlap between chunks
- `perform_analysis` (bool): Whether to analyze/summarize (default: False)
- `api_name` (str, optional): API provider for analysis
- `api_key` (str, optional): API key if not using config default
- `custom_prompt` (str, optional): Custom analysis prompt
- `system_prompt` (str, optional): System prompt for analysis
- `summarize_recursively` (bool): Recursive summarization (default: False)
- `overwrite` (bool): Overwrite existing content (default: False)

**Returns:**
Dictionary with:
- `success` (bool): Whether ingestion succeeded
- `media_id` (int): Database ID of ingested content
- `media_uuid` (str): UUID of ingested content
- `message` (str): Success/error message
- `file_path` (str): Path of processed file
- `media_type` (str): Detected media type
- `title` (str): Final title used
- `processing_details` (dict): Raw processing results

### `batch_ingest_files()`

Process multiple files with common options.

**Parameters:**
- `file_paths` (list): List of file paths to process
- `media_db` (MediaDatabase): Database instance
- `common_keywords` (list, optional): Keywords for all files
- `stop_on_error` (bool): Stop on first error (default: False)
- `**common_options`: Options applied to all files

**Returns:**
List of result dictionaries for each file.

### `ingest_directory()`

Scan and ingest all supported files in a directory.

**Parameters:**
- `directory_path` (str/Path): Directory to scan
- `media_db` (MediaDatabase): Database instance
- `recursive` (bool): Include subdirectories (default: False)
- `file_extensions` (list, optional): Limit to specific extensions
- `**common_options`: Options applied to all files

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
    perform_chunking=True,
    chunk_options={
        "method": "semantic",
        "chunk_size": 1000,
        "overlap": 200
    }
)
```

### Batch Processing with Common Settings

```python
files = [
    "/docs/report1.pdf",
    "/docs/report2.docx",
    "/docs/data.xlsx"
]

results = batch_ingest_files(
    file_paths=files,
    media_db=media_db,
    common_keywords=["2024", "quarterly"],
    perform_chunking=True,
    chunk_options={"method": "tokens", "chunk_size": 500}
)

# Check results
for r in results:
    print(f"{r['file_path']}: {'✓' if r['success'] else '✗'} {r['message']}")
```

### Directory Ingestion

```python
# Ingest all PDFs and EPUBs in a directory tree
results = ingest_directory(
    directory_path="/home/user/Documents",
    media_db=media_db,
    recursive=True,
    file_extensions=['.pdf', '.epub'],
    keywords=["archive", "2024"],
    perform_analysis=False  # Just store, don't analyze
)

print(f"Processed {len(results)} files")
successful = sum(1 for r in results if r['success'])
print(f"Success rate: {successful}/{len(results)}")
```

### Integration with Existing Code

```python
# In a larger application
class DocumentProcessor:
    def __init__(self, db_path: str):
        self.media_db = MediaDatabase(db_path, client_id="doc_processor")
    
    def process_upload(self, file_path: str, user_tags: list):
        """Process a user-uploaded document."""
        result = ingest_local_file(
            file_path=file_path,
            media_db=self.media_db,
            keywords=user_tags,
            perform_analysis=True,
            api_name=self.config.get('analysis_api')
        )
        
        if result['success']:
            self.notify_user(f"Document processed: {result['title']}")
            return result['media_id']
        else:
            self.handle_error(result['message'])
            return None
```

## Error Handling

The functions provide detailed error information:

```python
result = ingest_local_file(file_path="/path/to/file.pdf", media_db=db)

if not result['success']:
    if "not found" in result['message']:
        print("File doesn't exist")
    elif "Unsupported" in result['message']:
        print(f"File type not supported: {result['media_type']}")
    elif "Processing failed" in result['message']:
        # Check processing_details for more info
        print(f"Processing error: {result['processing_details'].get('error')}")
    elif "Database storage failed" in result['message']:
        print("Failed to save to database")
```

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

- The functions use the existing `MediaDatabase.add_media_with_keywords()` method
- Content is deduplicated by URL and content hash
- Use `overwrite=True` to update existing content
- All operations are transactional

## See Also

- `examples/local_file_ingestion_example.py` - Complete working examples
- Individual processor documentation in their respective modules
- Media database documentation in `DB/Client_Media_DB_v2.py`