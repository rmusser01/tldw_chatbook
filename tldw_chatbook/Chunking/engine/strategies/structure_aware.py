# strategies/structure_aware.py
"""
Structure-aware chunking strategy.
Preserves document structure including tables, headers, code blocks, and lists.
"""

import contextlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from loguru import logger

from ..base import BaseChunkingStrategy, ChunkMetadata, ChunkResult


class StructureType(Enum):
    """Types of document structures."""
    PARAGRAPH = "paragraph"
    HEADER = "header"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    LIST = "list"
    QUOTE = "quote"
    METADATA = "metadata"


class TableFormat(Enum):
    """Supported table formats."""
    MARKDOWN = "markdown"
    CSV = "csv"
    TSV = "tsv"
    HTML = "html"
    JSON = "json"
    PIPE_DELIMITED = "pipe"


@dataclass
class DocumentElement:
    """Represents a structural element in a document."""
    type: StructureType
    content: str
    level: Optional[int] = None  # For headers (h1=1, h2=2, etc.)
    language: Optional[str] = None  # For code blocks
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Table:
    """Represents a parsed table."""
    headers: list[str]
    rows: list[list[str]]
    format: TableFormat
    metadata: dict[str, Any] = None

    @property
    def num_columns(self) -> int:
        """Get number of columns."""
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)

    @property
    def num_rows(self) -> int:
        """Get number of data rows (excluding header)."""
        return len(self.rows)

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers and not self.rows:
            return ""

        # Use headers or generate default ones
        headers = self.headers if self.headers else [f"Col{i+1}" for i in range(self.num_columns)]

        # Build markdown table
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---" for _ in headers]) + "|")

        for row in self.rows:
            # Ensure row has correct number of columns
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")

        return "\n".join(lines)

    def to_text(self, style: str = "entity") -> str:
        """
        Convert table to text representation.

        Args:
            style: Serialization style ('entity', 'narrative', 'compact')
        """
        if style == "entity":
            return self._to_entity_text()
        elif style == "narrative":
            return self._to_narrative_text()
        else:
            return self._to_compact_text()

    def _to_entity_text(self) -> str:
        """Convert to entity-based text representation."""
        lines = []
        headers = self.headers if self.headers else [f"Col{i+1}" for i in range(self.num_columns)]

        for i, row in enumerate(self.rows):
            entity_parts = []
            for j, header in enumerate(headers):
                if j < len(row) and row[j]:
                    entity_parts.append(f"{header}: {row[j]}")

            if entity_parts:
                lines.append(f"Row {i+1}: " + "; ".join(entity_parts))

        return "\n".join(lines)

    def _to_narrative_text(self) -> str:
        """Convert to narrative text representation."""
        if not self.rows:
            return "Empty table"

        headers = self.headers if self.headers else [f"Col{i+1}" for i in range(self.num_columns)]

        narrative = f"A table with {len(headers)} columns ({', '.join(headers)}) "
        narrative += f"containing {len(self.rows)} rows of data. "

        # Describe first few rows
        for i, row in enumerate(self.rows[:3]):
            narrative += f"Row {i+1} contains: "
            parts = []
            for j, header in enumerate(headers):
                if j < len(row) and row[j]:
                    parts.append(f"{header} is '{row[j]}'")
            narrative += ", ".join(parts) + ". "

        if len(self.rows) > 3:
            narrative += f"... and {len(self.rows) - 3} more rows."

        return narrative

    def _to_compact_text(self) -> str:
        """Convert to compact text representation."""
        lines = []
        headers = self.headers if self.headers else []

        if headers:
            lines.append(" | ".join(headers))

        for row in self.rows:
            lines.append(" | ".join(row))

        return "\n".join(lines)


class StructureAwareChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text while preserving document structure.
    Handles tables, headers, code blocks, lists, and other structural elements.
    """

    def __init__(self, language: str = 'en'):
        """
        Initialize structure-aware chunking strategy.

        Args:
            language: Language code for text processing
        """
        super().__init__(language)

        # Regex patterns for structure detection
        self.patterns = {
            'markdown_header': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            # Broaden code fence support: ``` or ~~~, flexible language tags, optional newline before close
            # Groups: 1=fence, 2=language spec (may contain non-word chars), 3=body
            'code_block': re.compile(r'(?:\A|\n)(`{3,}|~{3,})([^\n]*)\n(.*?)(?:\n\1|\1)', re.DOTALL),
            'table_markdown': re.compile(r'^\|.*\|.*$', re.MULTILINE),
            'list_item': re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE),
            'numbered_list': re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE),
            'quote': re.compile(r'^>\s+(.+)$', re.MULTILINE),
        }

        logger.debug(f"StructureAwareChunkingStrategy initialized for language: {language}")

    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> list[str]:
        """
        Chunk text while preserving structure.

        Args:
            text: Text to chunk
            max_size: Maximum size per chunk (in elements, not characters)
            overlap: Number of elements to overlap between chunks
            **options: Additional options:
                - preserve_tables: Keep tables intact
                - preserve_code_blocks: Keep code blocks intact
                - preserve_headers: Include header hierarchy
                - table_serialization: How to serialize tables ('markdown', 'entity', 'narrative')
                - contextual_header_mode: 'none' (default), 'simple', or 'contextual' to prepend breadcrumbs
                - doc_title: Optional document title for breadcrumbs
                - folder_path: Optional folder path (e.g., "Workspace/Team/Project") for breadcrumbs
                - max_breadcrumb_levels: Limit number of header levels shown (default 6)

        Returns:
            List of text chunks preserving structure
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        # Align overlap semantics with other strategies to ensure progress
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        # Parse document structure
        elements = self._parse_document_structure(text, **options)

        if not elements:
            return []

        logger.debug(f"Parsed {len(elements)} structural elements")

        # Group elements into chunks
        chunks = self._group_elements_into_chunks(elements, max_size, overlap, **options)

        # Prepare global header index for breadcrumb computation
        global_headers: list[tuple[int, int, str]] = []  # (start, level, text)
        for e in elements:
            if e.type == StructureType.HEADER and isinstance(e.level, int):
                start_pos = e.metadata.get('start') if isinstance(e.metadata, dict) else None
                if isinstance(start_pos, int):
                    global_headers.append((start_pos, int(e.level), e.content.strip()))
        global_headers.sort(key=lambda t: t[0])

        # Convert chunks to text
        text_chunks = []
        for chunk_elements in chunks:
            # Determine chunk start based on earliest element position
            chunk_start = None
            for ce in chunk_elements:
                try:
                    s = ce.metadata.get('start') if isinstance(ce.metadata, dict) else None
                    if isinstance(s, int):
                        chunk_start = s if chunk_start is None else min(chunk_start, s)
                except Exception as metadata_error:
                    logger.debug("Structure-aware chunker failed to read element start metadata", exc_info=metadata_error)
            chunk_text = self._elements_to_text(
                chunk_elements,
                _global_headers=global_headers,
                _chunk_start=chunk_start if isinstance(chunk_start, int) else 0,
                **options
            )
            if chunk_text.strip():
                text_chunks.append(chunk_text)

        logger.debug(f"Created {len(text_chunks)} structure-aware chunks")
        return text_chunks

    def chunk_with_metadata(self,
                            text: str,
                            max_size: int,
                            overlap: int = 0,
                            **options) -> list[ChunkResult]:
        """Chunk text and return metadata using source spans from structural elements."""
        if not self.validate_parameters(text, max_size, overlap):
            return []
        if overlap >= max_size:
            logger.warning(f"Overlap ({overlap}) >= max_size ({max_size}), setting to max_size - 1")
            overlap = max_size - 1

        elements = self._parse_document_structure(text, **options)
        if not elements:
            return []

        chunks = self._group_elements_into_chunks(elements, max_size, overlap, **options)

        # Prepare global header index for breadcrumb computation (mirrors chunk())
        global_headers: list[tuple[int, int, str]] = []  # (start, level, text)
        for e in elements:
            if e.type == StructureType.HEADER and isinstance(e.level, int):
                start_pos = e.metadata.get('start') if isinstance(e.metadata, dict) else None
                if isinstance(start_pos, int):
                    global_headers.append((start_pos, int(e.level), e.content.strip()))
        global_headers.sort(key=lambda t: t[0])

        results: list[ChunkResult] = []
        total = len(chunks)
        for idx, chunk_elements in enumerate(chunks):
            chunk_start = None
            chunk_end = None
            for ce in chunk_elements:
                if not isinstance(ce, DocumentElement):
                    continue
                if isinstance(ce.metadata, dict):
                    s = ce.metadata.get('start')
                    e = ce.metadata.get('end')
                else:
                    s = None
                    e = None
                if isinstance(s, int):
                    chunk_start = s if chunk_start is None else min(chunk_start, s)
                if isinstance(e, int):
                    chunk_end = e if chunk_end is None else max(chunk_end, e)

            if chunk_start is None:
                chunk_start = 0
            if chunk_end is None:
                chunk_end = chunk_start
            if chunk_end < chunk_start:
                chunk_end = chunk_start
            with contextlib.suppress(Exception):
                chunk_end = self._expand_end_to_grapheme_boundary(text, chunk_end)

            chunk_text = self._elements_to_text(
                chunk_elements,
                _global_headers=global_headers,
                _chunk_start=int(chunk_start),
                **options,
            )
            if not chunk_text.strip():
                continue

            metadata = ChunkMetadata(
                index=idx,
                start_char=int(chunk_start),
                end_char=int(chunk_end),
                word_count=len(chunk_text.split()),
                language=self.language,
                overlap_with_previous=overlap if idx > 0 else 0,
                overlap_with_next=overlap if idx < total - 1 else 0,
                method='structure_aware',
                options={
                    'element_count': len(chunk_elements),
                },
            )
            results.append(ChunkResult(text=chunk_text, metadata=metadata))

        return results

    def _parse_document_structure(self, text: str, **options) -> list[DocumentElement]:
        """
        Parse document into structural elements.

        Args:
            text: Document text
            **options: Parsing options

        Returns:
            List of document elements
        """
        elements = []
        processed_ranges = []

        # Extract code blocks first (they can contain other patterns)
        if options.get('preserve_code_blocks', True):
            for match in self.patterns['code_block'].finditer(text):
                language = (match.group(2) or '').strip() or 'text'
                code_content = match.group(3)
                elements.append(DocumentElement(
                    type=StructureType.CODE_BLOCK,
                    content=code_content,
                    language=language,
                    metadata={'start': match.start(), 'end': match.end()}
                ))
                processed_ranges.append((match.start(), match.end()))

        # Extract tables
        if options.get('preserve_tables', True):
            tables = self._extract_tables(text, processed_ranges)
            elements.extend(tables)
            for elem in tables:
                if 'start' in elem.metadata and 'end' in elem.metadata:
                    processed_ranges.append((elem.metadata['start'], elem.metadata['end']))

        # Extract headers
        if options.get('preserve_headers', True):
            for match in self.patterns['markdown_header'].finditer(text):
                if not self._in_processed_range(match.start(), processed_ranges):
                    level = len(match.group(1))
                    header_text = match.group(2)
                    elements.append(DocumentElement(
                        type=StructureType.HEADER,
                        content=header_text,
                        level=level,
                        metadata={'start': match.start(), 'end': match.end()}
                    ))
                    processed_ranges.append((match.start(), match.end()))

        # Extract lists
        if options.get('preserve_lists', True):
            # Bullet lists
            for match in self.patterns['list_item'].finditer(text):
                if not self._in_processed_range(match.start(), processed_ranges):
                    elements.append(DocumentElement(
                        type=StructureType.LIST,
                        content=match.group(0),
                        metadata={'list_type': 'bullet', 'start': match.start(), 'end': match.end()}
                    ))
                    processed_ranges.append((match.start(), match.end()))

            # Numbered lists
            for match in self.patterns['numbered_list'].finditer(text):
                if not self._in_processed_range(match.start(), processed_ranges):
                    elements.append(DocumentElement(
                        type=StructureType.LIST,
                        content=match.group(0),
                        metadata={'list_type': 'numbered', 'start': match.start(), 'end': match.end()}
                    ))
                    processed_ranges.append((match.start(), match.end()))

        # Extract remaining paragraphs
        processed_ranges.sort()
        last_end = 0

        for start, end in processed_ranges:
            if start > last_end:
                raw_segment = text[last_end:start]
                if raw_segment and raw_segment.strip():
                    # Compute trimmed boundaries so offsets match original source text
                    leading_ws = len(raw_segment) - len(raw_segment.lstrip())
                    trailing_ws = len(raw_segment) - len(raw_segment.rstrip())
                    para_start = last_end + leading_ws
                    para_end = start - trailing_ws if trailing_ws else start
                    if para_end > para_start:
                        paragraph_text = text[para_start:para_end]
                        elements.append(DocumentElement(
                            type=StructureType.PARAGRAPH,
                            content=paragraph_text,
                            metadata={'start': para_start, 'end': para_end}
                        ))
            last_end = max(last_end, end)

        # Add final paragraph if any
        if last_end < len(text):
            raw_segment = text[last_end:]
            if raw_segment and raw_segment.strip():
                leading_ws = len(raw_segment) - len(raw_segment.lstrip())
                trailing_ws = len(raw_segment) - len(raw_segment.rstrip())
                para_start = last_end + leading_ws
                para_end = len(text) - trailing_ws if trailing_ws else len(text)
                if para_end > para_start:
                    paragraph_text = text[para_start:para_end]
                    elements.append(DocumentElement(
                        type=StructureType.PARAGRAPH,
                        content=paragraph_text,
                        metadata={'start': para_start, 'end': para_end}
                    ))

        # Sort elements by position
        elements.sort(key=lambda e: e.metadata.get('start', float('inf')))

        return elements

    def _extract_tables(self, text: str, processed_ranges: list[tuple[int, int]]) -> list[DocumentElement]:
        """
        Extract tables from text.

        Args:
            text: Document text
            processed_ranges: Already processed text ranges

        Returns:
            List of table elements
        """
        table_elements = []

        # Look for markdown tables
        lines = text.split('\n')
        # Precompute character offsets for line starts
        line_starts: list[int] = []
        pos = 0
        for idx, ln in enumerate(lines):
            line_starts.append(pos)
            # add 1 for the split '\n' separator except after last line
            pos += len(ln) + (0 if idx == len(lines) - 1 else 1)
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check if this looks like a markdown table
            line_start_char = line_starts[i] if i < len(line_starts) else 0
            if '|' in line and not self._in_processed_range(line_start_char, processed_ranges):
                # Try to parse as table
                table_lines = [line]
                j = i + 1

                # Look for separator line
                if j < len(lines) and re.match(r'^[\s\|:\-]+$', lines[j]):
                    table_lines.append(lines[j])
                    j += 1

                    # Collect data rows
                    while j < len(lines) and '|' in lines[j]:
                        table_lines.append(lines[j])
                        j += 1

                    # Parse the table
                    table = self._parse_markdown_table('\n'.join(table_lines))
                    if table:
                        table_text = table.to_markdown()
                        # Compute character offsets for the table block based on line indices
                        try:
                            start_char = line_starts[i] if i < len(line_starts) else 0
                            end_char = line_starts[j] if j < len(line_starts) else len(text)
                            with contextlib.suppress(Exception):
                                end_char = self._expand_end_to_grapheme_boundary(text, end_char)
                            if end_char < start_char:
                                end_char = start_char
                        except Exception:
                            start_char, end_char = (0, 0)
                        table_elements.append(DocumentElement(
                            type=StructureType.TABLE,
                            content=table_text,
                            metadata={
                                'table': table,
                                'format': 'markdown',
                                # Use character offsets for consistency with other elements
                                'start': start_char,
                                'end': end_char
                            }
                        ))
                        i = j
                        continue

            i += 1

        return table_elements

    def _parse_markdown_table(self, text: str) -> Optional[Table]:
        """Parse a markdown table."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

        if len(lines) < 2:
            return None

        # Parse header (preserve empty cells)
        header_line = lines[0]
        header_raw = header_line.strip().strip('|')
        headers = [cell.strip() for cell in header_raw.split('|')]

        # Skip separator line
        separator_idx = 1
        for i, line in enumerate(lines[1:], 1):
            if re.match(r'^[\s\|:\-]+$', line):
                separator_idx = i
                break

        # Parse rows
        rows = []
        for line in lines[separator_idx + 1:]:
            if '|' in line:
                row_raw = line.strip().strip('|')
                row = [cell.strip() for cell in row_raw.split('|')]
                rows.append(row)

        if headers or rows:
            return Table(headers=headers, rows=rows, format=TableFormat.MARKDOWN)

        return None

    def _in_processed_range(self, position: int, ranges: list[tuple[int, int]]) -> bool:
        """Check if position is in any processed range."""
        return any(start <= position < end for start, end in ranges)

    def _group_elements_into_chunks(self,
                                   elements: list[DocumentElement],
                                   max_size: int,
                                   overlap: int,
                                   **options) -> list[list[DocumentElement]]:
        """
        Group elements into chunks respecting structure.

        Args:
            elements: List of document elements
            max_size: Maximum elements per chunk
            overlap: Number of elements to overlap
            **options: Chunking options

        Returns:
            List of element groups (chunks)
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for element in elements:
            # Check if element should be kept intact
            keep_intact = False

            if element.type == StructureType.TABLE and options.get('preserve_tables', True) or element.type == StructureType.CODE_BLOCK and options.get('preserve_code_blocks', True):
                keep_intact = True

            # Estimate element size (simple count for now)
            element_size = 1

            # If adding element exceeds max_size and chunk is not empty
            if current_size + element_size > max_size and current_chunk:
                chunks.append(current_chunk)

                # Handle overlap
                if overlap > 0:
                    # Keep last 'overlap' elements for next chunk
                    current_chunk = current_chunk[-overlap:]
                    current_size = len(current_chunk)
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(element)
            current_size += element_size

            # If element must be kept intact and exceeds max_size alone
            if keep_intact and element_size > max_size and len(current_chunk) == 1:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _elements_to_text(self, elements: list[DocumentElement], **options) -> str:
        """
        Convert elements back to text.

        Args:
            elements: List of document elements
            **options: Serialization options

        Returns:
            Text representation
        """
        lines = []

        # Optionally prepend a computed contextual header derived from structure
        header_mode = str(options.get('contextual_header_mode', 'none')).lower()
        if header_mode in ('simple', 'contextual', 'header'):
            header_line = self._build_contextual_header(elements, options)
            if header_line:
                lines.append(header_line)
                lines.append("")

        for element in elements:
            if element.type == StructureType.HEADER:
                # Recreate markdown header
                level = element.level or 1
                lines.append(f"{'#' * level} {element.content}")
                lines.append("")  # Empty line after header

            elif element.type == StructureType.CODE_BLOCK:
                # Recreate code block
                language = element.language or ''
                lines.append(f"```{language}")
                lines.append(element.content)
                lines.append("```")
                lines.append("")

            elif element.type == StructureType.TABLE:
                # Serialize table based on option
                if 'table' in element.metadata:
                    table = element.metadata['table']
                    style = options.get('table_serialization', 'markdown')

                    if style == 'markdown':
                        lines.append(table.to_markdown())
                    elif style == 'entity':
                        lines.append(table.to_text('entity'))
                    elif style == 'narrative':
                        lines.append(table.to_text('narrative'))
                    else:
                        lines.append(element.content)
                else:
                    lines.append(element.content)
                lines.append("")

            elif element.type == StructureType.LIST:
                lines.append(element.content)

            elif element.type == StructureType.PARAGRAPH:
                lines.append(element.content)
                lines.append("")  # Empty line between paragraphs

            else:
                lines.append(element.content)

        return '\n'.join(lines).strip()

    def _build_contextual_header(self, elements: list[DocumentElement], options: dict[str, Any]) -> str:
        """Build contextual breadcrumbs for a chunk.

        Strategy:
        - Merge folder path and document title when available
        - Build header breadcrumbs using global header index up to the chunk start
        - Fallback to best in-chunk header if global headers unavailable
        """
        doc_title = options.get('doc_title')
        folder_path = options.get('folder_path')  # e.g., "Workspace/Team Docs/Project X"
        max_levels = int(options.get('max_breadcrumb_levels', 6))
        # Extract global headers and chunk start (provided by chunk())
        global_headers: list[tuple[int, int, str]] = options.get('_global_headers') or []
        chunk_start: int = options.get('_chunk_start') or 0

        # Compute header breadcrumb chain from global headers
        breadcrumb_sections: list[str] = []
        if global_headers:
            stack: list[tuple[int, str]] = []  # (level, text)
            for pos, lvl, text in global_headers:
                if pos >= chunk_start:
                    break
                # Maintain proper nesting: pop until top level < current
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, text))
            breadcrumb_sections = [t for (_lvl, t) in stack]
        else:
            # Fallback: use highest-level header within the chunk
            headers_in_chunk = [
                (e.level or 7, (e.content or '').strip())
                for e in elements
                if e.type == StructureType.HEADER and isinstance(e.content, str) and e.content.strip()
            ]
            headers_in_chunk.sort(key=lambda t: t[0])
            if headers_in_chunk:
                breadcrumb_sections = [h[1] for h in headers_in_chunk]

        # Trim breadcrumbs if too long
        if max_levels > 0 and len(breadcrumb_sections) > max_levels:
            breadcrumb_sections = breadcrumb_sections[-max_levels:]

        parts: list[str] = []
        # Folder path first if available
        if isinstance(folder_path, str) and folder_path.strip():
            parts.append(folder_path.strip())
        # Document title next
        if isinstance(doc_title, str) and doc_title.strip():
            parts.append(doc_title.strip())
        # Then section path
        if breadcrumb_sections:
            parts.extend(breadcrumb_sections)

        if not parts:
            return ""
        # Join into a single breadcrumb line
        return " > ".join(parts)
