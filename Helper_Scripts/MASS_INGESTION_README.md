# Mass Media Ingestion Guide

The `mass_ingest.py` script allows you to bulk import media files into the tldw_chatbook database without using the UI. This is perfect for processing large collections of documents, PDFs, e-books, and other supported files.

## Features

- **Bulk Processing**: Ingest entire directories of files at once
- **Recursive Scanning**: Optionally process subdirectories
- **File Type Filtering**: Process only specific types of files
- **Error Handling**: Continue processing even if some files fail
- **Progress Tracking**: Real-time progress bar and statistics
- **Analysis Support**: Optionally summarize/analyze content during ingestion
- **Keyword Management**: Add common keywords to all ingested files
- **Custom Chunking**: Configure how documents are split into chunks

## Supported File Types

- **PDF**: `.pdf`
- **Documents**: `.doc`, `.docx`, `.odt`, `.rtf`
- **E-books**: `.epub`, `.mobi`, `.azw`, `.azw3`, `.fb2`
- **XML**: `.xml`, `.rss`, `.atom`
- **Web**: `.html`, `.htm`
- **Text**: `.txt`, `.md`, `.markdown`, `.rst`, `.log`, `.csv`

## Quick Start

```bash
# Basic usage - ingest all supported files in a directory
python mass_ingest.py /path/to/documents

# Recursive ingestion (include subdirectories)
python mass_ingest.py /path/to/documents --recursive

# Ingest only PDFs and documents
python mass_ingest.py /path/to/documents --types pdf document

# Add keywords to all files
python mass_ingest.py /path/to/documents --keywords research 2024 important
```

## Advanced Usage

### Analysis and Summarization

To analyze/summarize content during ingestion (requires API key):

```bash
python mass_ingest.py /path/to/documents \
    --analyze \
    --api-name openai \
    --api-key YOUR_API_KEY
```

### Custom Chunking

Configure how documents are split into chunks:

```bash
python mass_ingest.py /path/to/documents \
    --chunk-method semantic \
    --chunk-size 1000 \
    --chunk-overlap 200
```

Available chunk methods:
- `semantic`: Context-aware chunking
- `tokens`: Split by token count
- `paragraphs`: Split by paragraphs
- `sentences`: Split by sentences  
- `words`: Split by word count

### Excluding Files

Skip files matching certain patterns:

```bash
python mass_ingest.py /path/to/documents \
    --exclude "*.tmp" "draft_*" "~*" ".*"
```

### Error Handling

By default, the script stops on the first error. To continue processing:

```bash
python mass_ingest.py /path/to/documents --continue-on-error
```

### Dry Run

Preview what would be ingested without actually processing:

```bash
python mass_ingest.py /path/to/documents --dry-run
```

### Custom Database Path

Use a specific database file:

```bash
python mass_ingest.py /path/to/documents --db-path /custom/path/to/media.db
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `directory` | Directory containing files to ingest (required) |
| `-r, --recursive` | Process subdirectories recursively |
| `-t, --types` | Only process specific file types |
| `-k, --keywords` | Keywords to add to all files |
| `-e, --exclude` | File patterns to exclude |
| `--analyze` | Perform analysis/summarization |
| `--api-name` | API provider for analysis |
| `--api-key` | API key for analysis |
| `--chunk-method` | Method for splitting documents |
| `--chunk-size` | Size of document chunks |
| `--chunk-overlap` | Overlap between chunks |
| `--continue-on-error` | Don't stop on failures |
| `--db-path` | Custom database path |
| `--dry-run` | Preview without processing |
| `--list-types` | Show supported file types |
| `-v, --verbose` | Detailed output |

## Output

The script provides:
- Real-time progress bar
- Success/failure status for each file
- Summary statistics
- Optional JSON results file (with `--verbose`)

## Examples

### Process Academic Papers

```bash
python mass_ingest.py ~/Documents/Research/Papers \
    --recursive \
    --types pdf \
    --keywords "academic" "research" "2024" \
    --chunk-method semantic \
    --analyze \
    --api-name anthropic
```

### Import E-book Collection

```bash
python mass_ingest.py ~/Books \
    --recursive \
    --types ebook \
    --keywords "fiction" "library" \
    --continue-on-error
```

### Process Work Documents

```bash
python mass_ingest.py ~/Work/Projects \
    --recursive \
    --types document pdf \
    --exclude "*.tmp" "~*" \
    --keywords "work" "project" \
    --chunk-size 500
```

## Tips

1. **Start Small**: Test with a few files first to ensure settings are correct
2. **Use Dry Run**: Preview what will be processed with `--dry-run`
3. **Monitor Progress**: Use `--verbose` for detailed feedback
4. **Organize First**: Structure your files in directories by type/topic
5. **Add Keywords**: Use meaningful keywords for better searchability
6. **Check Resources**: Large ingestions can use significant CPU/memory
7. **Backup Database**: Consider backing up your database before large imports

## Troubleshooting

- **Import Errors**: Ensure tldw_chatbook is installed or run from project directory
- **Database Errors**: Check database path and permissions
- **Memory Issues**: Process smaller batches or use fewer concurrent operations
- **API Errors**: Verify API key and check rate limits