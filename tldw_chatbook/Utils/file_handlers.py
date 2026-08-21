"""
File handlers for processing different file types in chat attachments.

This module provides a plugin-based architecture for handling various file types
that users can attach to chat messages. Each handler processes files differently:
- Images: Store as binary attachments
- Text/Code: Insert content directly into chat
- Documents: Extract and insert text
- Data files: Format and insert content
"""

import asyncio
import json
import mimetypes
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Literal, Union, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    pass


async def _extract_local_ingest_text(file_path: Path, *, kind_label: str) -> str:
    """Extract text via the shared local-ingestion parser (task-19576).

    This is the same extractor (``parse_local_file_for_ingest``) Library ▸
    Import uses -- PDF/document/ebook handlers used to return a placeholder
    telling the user to visit a "Media Ingestion tab" that was retired (its
    route now aliases to Library; see ``UI/Navigation/screen_registry.py``).

    Runs off the event loop: PDF/document/ebook extraction is CPU-bound and
    can easily exceed the repo's 100ms worker budget. No database write and
    no LLM analysis happen here (``perform_analysis`` defaults to False in
    ``parse_local_file_for_ingest``) -- this only reads and extracts text.
    The 100MB attachment cap enforced upstream by
    ``Chat/attachment_core.py``'s ``MAX_ATTACHMENT_BYTES`` (checked before
    any handler is dispatched) already bounds this, matching Library ▸
    Import's own extraction, which imposes no additional file-size cap of
    its own at this layer.

    Args:
        file_path: Path to the file to extract text from.
        kind_label: Human-readable file-type label used only when no text
            could be extracted (e.g. "PDF", "document", "ebook").

    Returns:
        Formatted content wrapped with the same
        ``--- Contents of ... ---``/``--- End of ... ---`` markers the other
        inline handlers in this module use, or an honest "no extractable
        text" notice (naming no destination) when extraction produced
        nothing.

    Raises:
        Exception: Whatever ``parse_local_file_for_ingest`` raises (missing
            optional dependency, corrupt file, etc.) -- callers catch this
            and surface an honest, non-placeholder error message.
    """
    from ..Local_Ingestion.local_file_ingestion import parse_local_file_for_ingest

    payload = await asyncio.to_thread(parse_local_file_for_ingest, str(file_path), {})
    content = (payload.get("content") or "").strip()
    if not content:
        return (
            f"[No extractable text found in {kind_label} file: {file_path.name}]"
        )
    return (
        f"--- Contents of {file_path.name} ---\n{content}\n"
        f"--- End of {file_path.name} ---"
    )


@dataclass
class ProcessedFile:
    """Result of processing a file attachment."""

    content: Optional[str] = None  # For text insertion into chat input
    attachment_data: Optional[bytes] = None  # For binary attachments (e.g., images)
    attachment_mime_type: Optional[str] = None  # MIME type for attachments
    display_name: str = ""  # Display name for UI
    insert_mode: Literal["inline", "attachment"] = "inline"  # How to handle the file
    file_type: str = "unknown"  # Type identifier for UI display


class FileHandler(ABC):
    """Base class for file handlers."""

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the given file."""
        pass

    @abstractmethod
    async def process(self, file_path: Path) -> ProcessedFile:
        """Process the file and return the result."""
        pass

    def get_mime_type(self, file_path: Path) -> str:
        """Get MIME type for the file."""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"


class ImageFileHandler(FileHandler):
    """Handler for image files - maintains existing functionality."""

    def can_handle(self, file_path: Path) -> bool:
        # Lazy import: attachment_core imports this module at module level
        # (same cycle-avoidance as the lazy import in process() below).
        from ..Chat.attachment_core import supported_image_formats

        return file_path.suffix.lower() in supported_image_formats()

    async def process(self, file_path: Path) -> ProcessedFile:
        """Process image file for attachment."""
        try:
            # Import the existing image handler
            from ..Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler

            # Use existing image processing logic
            image_data, mime_type = await ChatImageHandler.process_image_file(
                str(file_path)
            )

            return ProcessedFile(
                attachment_data=image_data,
                attachment_mime_type=mime_type,
                display_name=file_path.name,
                insert_mode="attachment",
                file_type="image",
            )
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            raise


class TextFileHandler(FileHandler):
    """Handler for plain text files."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".log", ".text", ".rst", ".textile"}
    MAX_FILE_SIZE = 1024 * 1024  # 1MB limit for text files

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def process(self, file_path: Path) -> ProcessedFile:
        """Read text file and prepare for insertion."""
        try:
            # Check file size
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                return ProcessedFile(
                    content=f"[File too large: {file_path.name} ({file_path.stat().st_size / 1024:.1f}KB)]",
                    display_name=file_path.name,
                    insert_mode="inline",
                    file_type="text",
                )

            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="replace")

            # Format for insertion
            formatted_content = f"--- Contents of {file_path.name} ---\n{content}\n--- End of {file_path.name} ---"

            return ProcessedFile(
                content=formatted_content,
                display_name=file_path.name,
                insert_mode="inline",
                file_type="text",
            )
        except Exception as e:
            logger.error(f"Failed to read text file {file_path}: {e}")
            return ProcessedFile(
                content=f"[Error reading file: {file_path.name}]",
                display_name=file_path.name,
                insert_mode="inline",
                file_type="text",
            )


class CodeFileHandler(FileHandler):
    """Handler for code files with syntax awareness."""

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".go": "go",
        ".rs": "rust",
        ".swift": "swift",
        ".kt": "kotlin",
        ".php": "php",
        ".r": "r",
        ".R": "r",
        ".m": "matlab",
        ".lua": "lua",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "zsh",
        ".fish": "fish",
        ".ps1": "powershell",
        ".sql": "sql",
        ".html": "html",
        ".htm": "html",
        ".xml": "xml",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".vue": "vue",
        ".svelte": "svelte",
    }

    MAX_FILE_SIZE = 512 * 1024  # 512KB limit for code files

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.LANGUAGE_MAP

    async def process(self, file_path: Path) -> ProcessedFile:
        """Read code file and format with language identifier."""
        try:
            # Check file size
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                return ProcessedFile(
                    content=f"[Code file too large: {file_path.name} ({file_path.stat().st_size / 1024:.1f}KB)]",
                    display_name=file_path.name,
                    insert_mode="inline",
                    file_type="code",
                )

            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="replace")

            # Get language for syntax highlighting
            language = self.LANGUAGE_MAP.get(file_path.suffix.lower(), "text")

            # Format as code block
            formatted_content = (
                f"```{language}\n# File: {file_path.name}\n{content}\n```"
            )

            return ProcessedFile(
                content=formatted_content,
                display_name=file_path.name,
                insert_mode="inline",
                file_type="code",
            )
        except Exception as e:
            logger.error(f"Failed to read code file {file_path}: {e}")
            return ProcessedFile(
                content=f"[Error reading code file: {file_path.name}]",
                display_name=file_path.name,
                insert_mode="inline",
                file_type="code",
            )


class DataFileHandler(FileHandler):
    """Handler for structured data files (JSON, YAML, CSV)."""

    SUPPORTED_EXTENSIONS = {".json", ".yaml", ".yml", ".csv", ".tsv"}
    MAX_FILE_SIZE = 256 * 1024  # 256KB limit for data files

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def process(self, file_path: Path) -> ProcessedFile:
        """Read and format structured data files."""
        try:
            # Check file size
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                return ProcessedFile(
                    content=f"[Data file too large: {file_path.name} ({file_path.stat().st_size / 1024:.1f}KB)]",
                    display_name=file_path.name,
                    insert_mode="inline",
                    file_type="data",
                )

            suffix = file_path.suffix.lower()

            if suffix == ".json":
                content = self._process_json(file_path)
            elif suffix in [".yaml", ".yml"]:
                content = self._process_yaml(file_path)
            elif suffix in [".csv", ".tsv"]:
                content = self._process_csv(file_path)
            else:
                content = file_path.read_text(encoding="utf-8", errors="replace")

            # Format for insertion
            formatted_content = f"--- Data from {file_path.name} ---\n{content}\n--- End of {file_path.name} ---"

            return ProcessedFile(
                content=formatted_content,
                display_name=file_path.name,
                insert_mode="inline",
                file_type="data",
            )
        except Exception as e:
            logger.error(f"Failed to process data file {file_path}: {e}")
            return ProcessedFile(
                content=f"[Error processing data file: {file_path.name}]",
                display_name=file_path.name,
                insert_mode="inline",
                file_type="data",
            )

    def _process_json(self, file_path: Path) -> str:
        """Process JSON file with pretty printing."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            return f"[Invalid JSON in {file_path.name}]"

    def _process_yaml(self, file_path: Path) -> str:
        """Process YAML file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return yaml.dump(data, default_flow_style=False, allow_unicode=True)
        except yaml.YAMLError:
            return f"[Invalid YAML in {file_path.name}]"

    def _process_csv(self, file_path: Path) -> str:
        """Process CSV/TSV file with basic formatting."""
        import csv

        try:
            delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","
            with open(file_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)

            if not rows:
                return "[Empty CSV file]"

            # Limit to first 20 rows for display
            if len(rows) > 20:
                rows = rows[:20] + [["... (truncated)"]]

            # Simple table format
            result = []
            for row in rows:
                result.append(" | ".join(str(cell) for cell in row))

            return "\n".join(result)
        except Exception as e:
            return f"[Error reading CSV: {e}]"


class PDFFileHandler(FileHandler):
    """Handler for PDF files, extracting real text via local ingestion."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def process(self, file_path: Path) -> ProcessedFile:
        """Extract PDF text via the shared local-ingestion parser (task-19576).

        This used to return a placeholder pointing at a retired "Media
        Ingestion tab" -- see `_extract_local_ingest_text`'s docstring.
        """
        try:
            content = await _extract_local_ingest_text(file_path, kind_label="PDF")
        except Exception as e:
            logger.error(f"Failed to extract PDF text from {file_path}: {e}")
            content = f"[Could not extract text from PDF file: {file_path.name} ({e})]"
        return ProcessedFile(
            content=content,
            display_name=file_path.name,
            insert_mode="inline",
            file_type="pdf",
        )


class DocumentFileHandler(FileHandler):
    """Handler for document files (Word, RTF, ODT), extracting real text via
    local ingestion."""

    SUPPORTED_EXTENSIONS = {".doc", ".docx", ".rtf", ".odt"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def process(self, file_path: Path) -> ProcessedFile:
        """Extract document text via the shared local-ingestion parser
        (task-19576). This used to return a placeholder pointing at a
        retired "Media Ingestion tab" -- see `_extract_local_ingest_text`'s
        docstring.
        """
        try:
            content = await _extract_local_ingest_text(
                file_path, kind_label="document"
            )
        except Exception as e:
            logger.error(f"Failed to extract document text from {file_path}: {e}")
            content = (
                f"[Could not extract text from document file: "
                f"{file_path.name} ({e})]"
            )
        return ProcessedFile(
            content=content,
            display_name=file_path.name,
            insert_mode="inline",
            file_type="document",
        )


class EbookFileHandler(FileHandler):
    """Handler for ebook files (EPUB, MOBI, AZW3), extracting real text via
    local ingestion."""

    SUPPORTED_EXTENSIONS = {".epub", ".mobi", ".azw", ".azw3", ".fb2"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def process(self, file_path: Path) -> ProcessedFile:
        """Extract ebook text via the shared local-ingestion parser
        (task-19576). This used to return a placeholder pointing at a
        retired "Media Ingestion tab" -- see `_extract_local_ingest_text`'s
        docstring.
        """
        try:
            content = await _extract_local_ingest_text(file_path, kind_label="ebook")
        except Exception as e:
            logger.error(f"Failed to extract ebook text from {file_path}: {e}")
            content = (
                f"[Could not extract text from ebook file: {file_path.name} ({e})]"
            )
        return ProcessedFile(
            content=content,
            display_name=file_path.name,
            insert_mode="inline",
            file_type="ebook",
        )


class PlaintextDatabaseHandler(FileHandler):
    """Handler for plaintext files that should be stored in the database."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".log", ".text", ".rst", ".textile"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit for database storage

    def can_handle(self, file_path: Path) -> bool:
        # Check if this is a plaintext file AND user wants database storage
        # For now, we'll handle this based on file size - large files go to DB
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False
        # Store files larger than 100KB in database
        return file_path.stat().st_size > 100 * 1024

    async def process(self, file_path: Path) -> ProcessedFile:
        """Process plaintext file for database ingestion.

        Task-19576: both branches used to name a "Media Ingestion tab" that
        no longer exists (its route now aliases to Library; see
        `UI/Navigation/screen_registry.py`). Unlike the PDF/document/ebook
        handlers above, this is a deliberate size-based deferral, not a
        missing-capability placeholder: reading text out of a plaintext
        file needs no extraction step, but inlining a many-hundred-KB (or
        multi-MB) file straight into a chat turn's context is a real
        cost/context-window concern of its own, so this still defers to
        Library's RAG-searchable ingestion instead of dumping raw text
        inline -- only the (now retired) destination name is fixed.
        """
        try:
            # Check file size
            if file_path.stat().st_size > self.MAX_FILE_SIZE:
                return ProcessedFile(
                    content=f"[File too large: {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)]\n"
                    f"To ingest this file for search, add it via the Library tab.",
                    display_name=file_path.name,
                    insert_mode="inline",
                    file_type="text",
                )

            # For large text files, suggest using Library's RAG-searchable
            # ingestion rather than inlining the whole file into this chat turn.
            return ProcessedFile(
                content=f"[Large Text File: {file_path.name} ({file_path.stat().st_size / 1024:.1f}KB)]\n"
                f"To process this file for RAG search, add it via the Library tab.",
                display_name=file_path.name,
                insert_mode="inline",
                file_type="text",
            )
        except Exception as e:
            logger.error(f"Failed to check text file {file_path}: {e}")
            return ProcessedFile(
                content=f"[Error checking text file: {file_path.name}]",
                display_name=file_path.name,
                insert_mode="inline",
                file_type="text",
            )


class DefaultFileHandler(FileHandler):
    """Fallback handler for unsupported file types."""

    def can_handle(self, file_path: Path) -> bool:
        """This handler can handle any file as a fallback."""
        return True

    async def process(self, file_path: Path) -> ProcessedFile:
        """Just reference the file by name and path."""
        self._reject_config_excluded_image(file_path)
        file_size = file_path.stat().st_size
        size_str = self._format_size(file_size)

        content = (
            f"[File: {file_path.name} ({size_str}) - {self.get_mime_type(file_path)}]"
        )

        return ProcessedFile(
            content=content,
            display_name=file_path.name,
            insert_mode="inline",
            file_type="file",
        )

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"

    @staticmethod
    def _reject_config_excluded_image(file_path: Path) -> None:
        """Reject known-image files that config excluded, instead of inlining
        a ``[File: …]`` placeholder that reads as success while sending
        useless text (legacy behavior was an explicit error).

        Reaching this fallback with a known image suffix already implies the
        effective ``[chat.images].supported_formats`` list excluded it —
        ImageFileHandler (first in the registry chain) claims every effective
        format.

        Raises:
            ValueError: When the extension is a known image format excluded
                by the current configuration.
        """
        # Lazy import: attachment_core imports this module at module level.
        from ..Chat.attachment_core import (
            DEFAULT_SUPPORTED_IMAGE_FORMATS,
            supported_image_formats,
        )

        suffix = file_path.suffix.lower()
        if suffix not in DEFAULT_SUPPORTED_IMAGE_FORMATS:
            return
        effective = supported_image_formats()
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported formats: {', '.join(effective)} "
            "([chat.images].supported_formats controls this list)"
        )


class FileHandlerRegistry:
    """Registry for managing file handlers."""

    def __init__(self):
        self.handlers: List[FileHandler] = [
            ImageFileHandler(),
            PDFFileHandler(),
            DocumentFileHandler(),
            EbookFileHandler(),
            PlaintextDatabaseHandler(),  # For large text files
            TextFileHandler(),  # For small text files (inline)
            CodeFileHandler(),
            DataFileHandler(),
            DefaultFileHandler(),  # Must be last as it handles everything
        ]

    async def process_file(self, file_path: Union[str, Path]) -> ProcessedFile:
        """Process a file using the appropriate handler."""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return ProcessedFile(
                content=f"[File not found: {file_path.name}]",
                display_name=file_path.name,
                insert_mode="inline",
                file_type="error",
            )

        # Find the first handler that can process this file
        for handler in self.handlers:
            if handler.can_handle(file_path):
                logger.debug(f"Using {handler.__class__.__name__} for {file_path.name}")
                return await handler.process(file_path)

        # This should never happen since DefaultFileHandler handles everything
        logger.error(f"No handler found for {file_path}")
        return ProcessedFile(
            content=f"[No handler for: {file_path.name}]",
            display_name=file_path.name,
            insert_mode="inline",
            file_type="error",
        )


# Global registry instance
file_handler_registry = FileHandlerRegistry()
