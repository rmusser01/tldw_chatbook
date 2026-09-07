# Chat File Upload System Documentation

> **Verified against the Console (task-577, 2026-07-25).** This doc originally
> described `ChatWindowEnhanced`'s upload UI and `chat_events.py`'s send path.
> Both were retired in task-577 (never-instantiated legacy window family and
> its dead send pipeline). The file-type handling layer (`file_handlers.py`)
> and the shared processing layer (`Chat/attachment_core.py`) it describes are
> unaffected and are the layer the live Console composer uses today; the
> UI-facing sections below have been rewritten to match.

## Overview

The Chat File Upload System in tldw_chatbook provides a flexible, extensible
mechanism for attaching and processing various file types in Console chat
conversations. Images are staged as binary attachments; text, code, data,
document, and e-book files are either inlined into the composer or referenced
for full ingestion, through a plugin-based handler architecture.

## Table of Contents

1. [Architecture](#architecture)
2. [Supported File Types](#supported-file-types)
3. [User Guide](#user-guide)
4. [Developer Guide](#developer-guide)
5. [File Handler System](#file-handler-system)
6. [Implementation Details](#implementation-details)
7. [Extending the System](#extending-the-system)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Architecture

The file upload system consists of three layers:

```
┌───────────────────────────────┐
│   UI Layer                    │
│ (UI/Screens/chat_screen.py)   │
│  - Console composer "Attach"  │
│  - Paste/drop path detection  │
│  - Alt+V clipboard image grab │
│  - Staged-attachment chips    │
└──────────────┬─────────────────┘
               │
┌──────────────▼─────────────────┐
│   Shared Processing Layer      │
│ (Chat/attachment_core.py)      │
│  - Validation, size caps       │
│  - Vision-model gating         │
│  - PendingAttachment staging   │
└──────────────┬─────────────────┘
               │
┌──────────────▼─────────────────┐
│   File Handler Layer           │
│ (Utils/file_handlers.py)       │
│  - Handler registry            │
│  - Type-specific processing    │
└──────────────┬─────────────────┘
               │
┌──────────────▼─────────────────┐
│   Persistence Layer            │
│ (Chat/console_chat_store.py +  │
│  DB/ChaChaNotes_DB.py)         │
│  - Position 0: messages.image_data/image_mime_type
│  - Positions >= 1: message_attachments table
└─────────────────────────────────┘
```

### Design Principles

1. **One pipeline**: `attachment_core.py` is the single validation/processing/
   vision-gating seam the Console composer calls into — no UI-specific
   duplication.
2. **Extensibility**: Easy to add new file type handlers to the registry.
3. **Type-Specific Processing**: Different file types are handled appropriately.
4. **User Experience**: Clear feedback and intuitive behavior (staged-chip
   indicators, per-message attachment cap).
5. **Security**: Path validation (`is_safe_path`), file-size caps, and
   safe parsing (`yaml.safe_load`, `json.load`) throughout.

## Supported File Types

### Images (Attachments)
- **Extensions**: configurable via `[chat.images].supported_formats`, default
  `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`, `.svg`
  (`.svg` is dropped automatically when `cairosvg` isn't installed)
- **Behavior**: Staged as a binary attachment, sent with the message
- **Size Limit**: `[chat.images].max_size_mb`, default 10 MB
- **Processing**: PIL-validated, resized to `[chat.images].resize_max_dimension`
  (default 2048px), base64-encoded for the provider call

### PDFs, Word/RTF/ODT documents, e-books (Reference)
- **Extensions**: `.pdf`; `.doc`, `.docx`, `.rtf`, `.odt`; `.epub`, `.mobi`,
  `.azw`, `.azw3`, `.fb2`
- **Behavior**: Inserted as a placeholder pointing to the Media Ingestion tab
  — these formats are not text-extracted inline
- **Size Limit**: none at this layer (governed by the general attachment cap)

### Large Plaintext Files (Reference)
- **Extensions**: `.txt`, `.md`, `.log`, `.text`, `.rst`, `.textile` **over
  100KB**
- **Behavior**: Inserted as a placeholder pointing to the Media Ingestion tab
  (files under 100KB fall through to the inline Text handler below)
- **Size Limit**: 10MB hard cap; larger files show a "too large" placeholder

### Text Files (Inline)
- **Extensions**: `.txt`, `.md`, `.log`, `.text`, `.rst`, `.textile`, **100KB
  or smaller**
- **Behavior**: Content inserted directly into the composer, wrapped with
  filename markers
- **Size Limit**: 1MB (hardcoded in `TextFileHandler`)

### Code Files (Inline)
- **Extensions**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`, `.cs`,
  `.rb`, `.go`, `.rs`, and more (see `CodeFileHandler.LANGUAGE_MAP`)
- **Behavior**: Content inserted as a fenced code block with a language tag
- **Size Limit**: 512KB (hardcoded in `CodeFileHandler`)

### Data Files (Inline)
- **Extensions**: `.json`, `.yaml`, `.yml`, `.csv`, `.tsv`
- **Behavior**: Content formatted (pretty-printed JSON/YAML, tabular CSV/TSV
  capped at 20 rows) and inserted
- **Size Limit**: 256KB (hardcoded in `DataFileHandler`)

### Other Files (Reference)
- **Extensions**: anything not matched above
- **Behavior**: File info (name, size, MIME type) inserted as reference text

## User Guide

### Attaching Files (Console)

1. Click **Attach** on the Console composer (or use the paste/drop paths
   below) — up to 5 attachments per message
   (`console_chat_store.MAX_PENDING_ATTACHMENTS`)
2. Pick a file from the picker dialog. The file is processed based on its
   type and staged as a chip showing `<name> · <size>`
3. Compose your message and send: staged attachments ride along with the
   message; inline text/code/data insertions become part of the message text

### Paste and drag-drop

Terminals don't deliver real drag-and-drop or clipboard-image paste events,
so the Console detects two proxies instead:

- **Drop or paste a file path** as text — the composer recognizes it and
  attaches the referenced file (`Chat/console_paste_attach.py`)
- **Alt+V** grabs an image directly from the OS clipboard (macOS/Windows via
  `PIL.ImageGrab`; unavailable on most Linux setups)

### Clearing Attachments

- Use the **Clear** control (`#console-clear-attachment`) on a staged chip to
  drop it before sending
- Inline insertions (text/code/data) are edited or removed like any other
  composer text

## Developer Guide

### File Handler System

The system uses a plugin architecture where each file type has a dedicated
handler:

```python
from abc import ABC, abstractmethod

class FileHandler(ABC):
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the file."""

    @abstractmethod
    async def process(self, file_path: Path) -> ProcessedFile:
        """Process the file and return result."""
```

### ProcessedFile Structure

```python
@dataclass
class ProcessedFile:
    content: Optional[str] = None  # For inline insertion
    attachment_data: Optional[bytes] = None  # For binary attachments
    attachment_mime_type: Optional[str] = None  # MIME type
    display_name: str = ""  # UI display name
    insert_mode: Literal["inline", "attachment"] = "inline"
    file_type: str = "unknown"  # Type identifier
```

### Handler Registration

Handlers are registered in priority order (`Utils/file_handlers.py`,
`FileHandlerRegistry.__init__`); the first handler whose `can_handle()`
returns `True` wins:

```python
class FileHandlerRegistry:
    def __init__(self):
        self.handlers = [
            ImageFileHandler(),
            PDFFileHandler(),
            DocumentFileHandler(),
            EbookFileHandler(),
            PlaintextDatabaseHandler(),  # large (>100KB) plaintext -> Ingestion pointer
            TextFileHandler(),           # small plaintext -> inline
            CodeFileHandler(),
            DataFileHandler(),
            DefaultFileHandler(),        # catch-all, must be last
        ]
```

## Implementation Details

### Staging on the Console session

`Chat/attachment_core.py` turns a raw path or clipboard bytes into a
`PendingAttachment` staged on the active Console session
(`ConsoleChatStore.pending_attachments`):

```python
@dataclass
class PendingAttachment:
    file_path: str
    display_name: str
    file_type: str
    insert_mode: Literal["inline", "attachment"]
    data: bytes | None = None
    mime_type: str | None = None
    text_content: str | None = None
    original_size: int = 0
    processed_size: int = 0
```

### Event Flow

1. **File Selection** — composer "Attach" button opens a file picker built
   from `attachment_filter_specs()`, or the paste/drop/Alt+V paths above
   detect a candidate directly.
2. **File Processing** — `attachment_core.process_attachment_path()` (file)
   or `process_attachment_bytes()` (clipboard image) validates the path/size
   and dispatches through `file_handler_registry.process_file()`.
3. **Staging** — the resulting `PendingAttachment` is appended to the
   session's pending list (capped at `MAX_PENDING_ATTACHMENTS`); the composer
   shows it as a labeled chip.
4. **Sending** — on send, staged attachments are folded into the outgoing
   message: position 0 mirrors the legacy scalar image fields, positions
   ≥ 1 become `message_attachments` rows (see Database Storage below); the
   send is blocked with a user-facing notice
   (`attachment_core.vision_block_reason`) if the selected model can't accept
   images.

### Database Storage

Attachments persist across two places in `ChaChaNotes_DB.py`, for backward
compatibility with the original single-image schema:

- **Position 0** (the first/only attachment): `messages.image_data` /
  `messages.image_mime_type` scalar columns (unchanged since the original
  image-only design).
- **Positions ≥ 1** (additional attachments on the same message): the
  `message_attachments` table (added in the schema v18→v19 migration),
  keyed by `message_id` + `position`, storing `data`, `mime_type`, and
  `display_name`.

```sql
CREATE TABLE IF NOT EXISTS message_attachments(
    message_id   TEXT NOT NULL,
    position     INTEGER NOT NULL,
    data         BLOB,
    mime_type    TEXT,
    display_name TEXT,
    ...
);
```

## Extending the System

### Adding a New File Handler

1. **Create Handler Class**
   ```python
   class ArchiveFileHandler(FileHandler):
       SUPPORTED_EXTENSIONS = {'.zip', '.tar', '.tar.gz'}

       def can_handle(self, file_path: Path) -> bool:
           return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

       async def process(self, file_path: Path) -> ProcessedFile:
           listing = await list_archive_contents(file_path)
           return ProcessedFile(
               content=f"--- Archive: {file_path.name} ---\n{listing}\n---",
               display_name=file_path.name,
               insert_mode="inline",
               file_type="archive",
           )
   ```

2. **Register Handler** — insert it at the priority position that matches
   its specificity (before `DefaultFileHandler`, which must stay last):
   ```python
   # In FileHandlerRegistry.__init__()
   self.handlers = [
       ImageFileHandler(),
       ArchiveFileHandler(),  # add here
       PDFFileHandler(),
       # ...
   ]
   ```

3. **Update the picker filters** (optional) — extend
   `_NON_IMAGE_FILTER_SPECS` / `_ALL_FILES_NON_IMAGE_PATTERNS` in
   `Chat/attachment_core.py` so the new extension shows up in the Console's
   file picker.

## Configuration

The only user-facing configuration today is `[chat.images]`:

```toml
[chat.images]
show_attach_button = true               # show/hide the Attach control
max_size_mb = 10.0                       # image byte cap
resize_max_dimension = 2048              # resize bound (px)
supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"]
```

The per-type size limits for text (1MB), code (512KB), and data (256KB)
files are handler constants (`Utils/file_handlers.py`), not configurable —
there is no `[chat.uploads]` config section.

## API Reference

### `Chat/attachment_core.py`

```python
async def process_attachment_path(file_path: str, *, allowed_root: str | None = None) -> PendingAttachment:
    """Validate, process, and normalize a file into a PendingAttachment."""

async def process_attachment_bytes(data: bytes, *, display_name: str, mime_type: str = "image/png") -> PendingAttachment:
    """Build an image PendingAttachment from raw bytes (clipboard path)."""

def attachment_filter_specs() -> tuple[tuple[str, str], ...]:
    """Picker filter rows, image patterns derived from the effective formats."""

def supported_image_formats() -> tuple[str, ...]:
    """Effective image extension allowlist from [chat.images].supported_formats."""

def vision_block_reason(provider: str, model: str | None, *, is_capable=None) -> str | None:
    """User-facing blocked-send copy when the model can't accept images."""
```

### `Utils/file_handlers.py`

```python
def can_handle(self, file_path: Path) -> bool:
    """Returns True if handler can process this file type."""

async def process(self, file_path: Path) -> ProcessedFile:
    """Process file and return ProcessedFile result."""

async def process_file(self, file_path: Union[str, Path]) -> ProcessedFile:
    """Route file to appropriate handler and process it (FileHandlerRegistry)."""
```

## Troubleshooting

### Common Issues

1. **File Not Processing**
   - Check file size limits
   - Verify file extension is supported
   - Check file permissions

2. **Content Not Inserting**
   - Ensure the composer is not at maximum length
   - Check for encoding issues with non-UTF8 files

3. **Images Not Attaching**
   - Verify the image format is in the effective `supported_formats` list
   - Check image size against `max_size_mb`
   - Ensure a vision-capable model is selected (see `vision_block_reason`)

### Debug Logging

Enable debug logging to troubleshoot:

```python
# Check logs for:
"Using {HandlerName} for {filename}"
"Processing file attachment: {file_path}"
```

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "File too large" | Exceeds size limit | Use smaller file or increase the relevant cap |
| "Invalid JSON/YAML in ..." | Data file malformed | Fix file syntax |
| "Unsupported image format" | Extension excluded by `[chat.images].supported_formats` | Add the extension to config or pick another file |
| "File path is outside allowed directories" | Path validation rejected the file | Attach a file under your home directory (or the configured allowed root) |

## Security Considerations

1. **File Size Limits**: Prevent memory exhaustion (per-type caps +
   `MAX_ATTACHMENT_BYTES` overall)
2. **Path Validation**: `is_safe_path` blocks directory traversal
3. **Content Validation**: JSON/YAML parsing in safe mode
4. **Binary Validation**: PIL-based image format verification
5. **Encoding Safety**: Text files decoded with `errors="replace"`

## Conclusion

The Chat File Upload System provides a robust, extensible foundation for
handling diverse file types in Console chat conversations. Its plugin
architecture (`Utils/file_handlers.py`) and shared processing seam
(`Chat/attachment_core.py`) keep type-specific logic isolated from the UI
layer, which today is the native Console composer in
`UI/Screens/chat_screen.py`.
