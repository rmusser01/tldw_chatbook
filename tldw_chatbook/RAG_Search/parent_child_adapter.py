"""
Parent/child chunking adapter over the vendored engine's hierarchical path.

Phase B (chunking-engine-parity, task 8, Q5 ruling): the home-grown
structure-aware chunking that used to live in
``RAG_Search.enhanced_chunking_service`` is retired. The engine's
``structure_aware`` strategy driven through ``Chunker.chunk_text_hierarchical_flat``
is now the only structure-aware implementation. This adapter is the seam
that keeps the legacy consumers working:

* :func:`chunk_with_parent_retrieval` preserves the legacy
  ``EnhancedChunkingService.chunk_with_parent_retrieval`` return shape
  (``{"chunks": [...], "parent_chunks": [...], "metadata": {...}}`` with
  ``parent_chunk_index`` references) while deriving everything from the
  engine's hierarchical output.
* :func:`chunk_text_with_structure` preserves the legacy
  ``List[StructuredChunk]`` return shape the chunk preview modal reads.

Parent/child derivation from the engine's flat hierarchical output:

* Children are the flat chunks (per-section element groups, document order).
  Each carries ``ancestry_titles`` (its section path) in its metadata.
* Consecutive children sharing the same top-level ancestry title belong to
  the same top-level section; that run forms a parent. Long runs are split
  so a parent holds at most ``max_size * parent_size_multiplier`` elements
  (mirroring the legacy "parent is N x retrieval size" intent).
* A parent's text is the original-text slice spanning its children, so a
  parent always contains its children (the containment the legacy mapping
  loop approximated by scanning for a covering parent chunk).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from tldw_chatbook.Chunking.engine import Chunker

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkType",
    "StructuredChunk",
    "chunk_with_parent_retrieval",
    "chunk_text_with_structure",
]

# Defaults mirror the legacy EnhancedChunkingService signatures.
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_PARENT_SIZE_MULTIPLIER = 3

# The engine's structure_aware strategy is the single structure-aware
# implementation (Q5 ruling); the hierarchical flat wrapper provides the
# section ancestry metadata this adapter derives parents from.
_STRUCTURE_METHOD = "structure_aware"


class ChunkType(Enum):
    """Types of chunks based on document structure (legacy vocabulary)."""

    PARAGRAPH = "paragraph"
    SECTION = "section"
    SUBSECTION = "subsection"
    LIST = "list"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    HEADER = "header"
    TEXT = "text"  # Generic text


# Engine-normalized chunk types -> legacy ChunkType members.
_ENGINE_CHUNK_TYPE_MAP: Dict[str, ChunkType] = {
    "heading": ChunkType.HEADER,
    "header": ChunkType.HEADER,
    "list": ChunkType.LIST,
    "code": ChunkType.CODE_BLOCK,
    "code_block": ChunkType.CODE_BLOCK,
    "table": ChunkType.TABLE,
    "quote": ChunkType.QUOTE,
    "footnote": ChunkType.FOOTNOTE,
    "section": ChunkType.SECTION,
    "subsection": ChunkType.SUBSECTION,
    "paragraph": ChunkType.PARAGRAPH,
    "text": ChunkType.TEXT,
}


@dataclass
class StructuredChunk:
    """Enhanced chunk with structural information (legacy attribute names).

    Kept attribute-compatible with the retired implementation so consumers
    such as ``Widgets/chunk_preview_modal.py`` (``chunk_index``,
    ``word_count``, ``char_count``, ``chunk_type.value``, ``metadata``)
    and ``Tests/test_enhanced_rag.py`` (``level``, ``start_char``,
    ``end_char``, ``parent_index``, ``children_indices``) keep working.
    """

    text: str
    start_char: int
    end_char: int
    chunk_index: int
    chunk_type: ChunkType
    level: int = 0  # Hierarchical level (0 = root)
    parent_index: Optional[int] = None
    parent_id: Optional[str] = None
    children_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Calculate word count."""
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        """Calculate character count."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the legacy dictionary format."""
        return {
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type.value,
            "level": self.level,
            "parent_index": self.parent_index,
            "children_indices": self.children_indices,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }


_chunker: Optional[Chunker] = None
_chunker_lock = threading.Lock()


def _get_chunker() -> Chunker:
    """Lazily create the shared engine chunker instance."""
    global _chunker
    if _chunker is None:
        with _chunker_lock:
            if _chunker is None:
                _chunker = Chunker()
    return _chunker


def _resolve_sizes(
    max_size: Optional[int],
    overlap: Optional[int],
    opts: Dict[str, Any],
) -> Tuple[int, int]:
    """Resolve engine-style sizes, honouring the legacy kwarg aliases.

    The legacy service used ``chunk_size``/``chunk_overlap``; the adapter's
    primary signature uses the engine's ``max_size``/``overlap``. Both are
    accepted (``max_size``/``overlap`` win when both are supplied).
    """
    legacy_size = opts.pop("chunk_size", None)
    legacy_overlap = opts.pop("chunk_overlap", None)
    if max_size is None:
        max_size = legacy_size if legacy_size is not None else DEFAULT_CHUNK_SIZE
    if overlap is None:
        overlap = (
            legacy_overlap if legacy_overlap is not None else DEFAULT_CHUNK_OVERLAP
        )
    try:
        return int(max_size), int(overlap)
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def _map_chunk_type(raw: Any) -> ChunkType:
    """Map an engine chunk type (normalized string or enum) to ChunkType."""
    if raw is None:
        return ChunkType.TEXT
    value = getattr(raw, "value", raw)
    try:
        key = str(value).strip().lower()
    except Exception:  # pragma: no cover - defensive
        return ChunkType.TEXT
    return _ENGINE_CHUNK_TYPE_MAP.get(key, ChunkType.TEXT)


def _engine_flat(
    text: str, max_size: int, overlap: int
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the engine's hierarchical structure_aware path and flatten it.

    Returns ``(clean_text, flat)``: the engine sanitizes its input once
    (length-preserving — nulls/control/bidi-override chars become spaces)
    and every chunk offset refers to that sanitized text, so parent slices
    must be taken from the same text — slicing the raw input would
    re-introduce exactly the characters the engine neutralized in the
    children (Qodo PR #1852). ``_sanitize_input`` is a private engine seam
    (like the shim's tokenizer probe): re-check at next sync.
    """
    chunker = _get_chunker()
    flat = chunker.chunk_text_hierarchical_flat(
        text, method=_STRUCTURE_METHOD, max_size=max_size, overlap=overlap
    )
    try:
        clean_text = chunker._sanitize_input(text, suppress_security_log=True)
    except Exception:  # pragma: no cover - defensive: raw slices, as before
        clean_text = text
    return clean_text, flat


def _build_children(flat: List[Dict[str, Any]]) -> List[StructuredChunk]:
    """Convert engine flat chunks to StructuredChunks in document order."""
    children: List[StructuredChunk] = []
    fallback_pos = 0
    for i, item in enumerate(flat):
        if isinstance(item, dict):
            item_text = item.get("text")
            item_text = item_text if isinstance(item_text, str) else str(item_text)
            md = item.get("metadata")
            md = dict(md) if isinstance(md, dict) else {}
        else:
            item_text = str(item)
            md = {}
        start = md.get("start_offset")
        end = md.get("end_offset")
        if not isinstance(start, int) or not isinstance(end, int):
            # Defensive: structure_aware always carries offsets, but never
            # crash if a future engine revision drops them.
            start, end = fallback_pos, fallback_pos + len(item_text)
        fallback_pos = end
        ancestry = md.get("ancestry_titles")
        ancestry = ancestry if isinstance(ancestry, list) else []
        children.append(
            StructuredChunk(
                text=item_text,
                start_char=start,
                end_char=end,
                chunk_index=i,
                chunk_type=_map_chunk_type(md.get("chunk_type")),
                level=len(ancestry),
                metadata=md,
            )
        )
    return children


def _group_into_parents(
    text: str,
    children: List[StructuredChunk],
    max_size: int,
    parent_size_multiplier: int,
) -> List[StructuredChunk]:
    """Derive parent chunks from the children's top-level section ancestry.

    Consecutive children sharing the same top-level ancestry title form one
    top-level section run; each run becomes one parent, split when a parent
    would exceed ``max_size * parent_size_multiplier`` grouped elements.
    Children are mutated in place with their parent linkage.
    """
    # Partition children into top-level section runs (document order).
    runs: List[Tuple[Optional[str], List[StructuredChunk]]] = []
    for child in children:
        ancestry = child.metadata.get("ancestry_titles")
        ancestry = ancestry if isinstance(ancestry, list) else []
        top_title = ancestry[0] if ancestry else None
        if not isinstance(top_title, str):
            top_title = None
        if runs and runs[-1][0] == top_title:
            runs[-1][1].append(child)
        else:
            runs.append((top_title, [child]))

    # Element budget per parent (0 disables splitting: one parent per run).
    budget = 0
    try:
        if int(max_size) > 0 and int(parent_size_multiplier) > 0:
            budget = int(max_size) * int(parent_size_multiplier)
    except (TypeError, ValueError):
        budget = 0

    # Split each run into parent groups honouring the element budget.
    groups: List[Tuple[Optional[str], List[StructuredChunk]]] = []
    for title, run in runs:
        current: List[StructuredChunk] = []
        current_elements = 0
        for child in run:
            elements = child.metadata.get("grouped_elements")
            if not isinstance(elements, int) or elements <= 0:
                elements = 1
            if budget > 0 and current and current_elements + elements > budget:
                groups.append((title, current))
                current = []
                current_elements = 0
            current.append(child)
            current_elements += elements
        if current:
            groups.append((title, current))

    parents: List[StructuredChunk] = []
    for parent_index, (title, group) in enumerate(groups):
        start = min(c.start_char for c in group)
        end = max(c.end_char for c in group)
        if 0 <= start <= end <= len(text):
            parent_text = text[start:end]
        else:  # pragma: no cover - offsets out of range
            parent_text = "".join(c.text for c in group)
        metadata: Dict[str, Any] = {"element_type": "section"}
        if title:
            metadata["section_title"] = title
        parent = StructuredChunk(
            text=parent_text,
            start_char=start,
            end_char=end,
            chunk_index=parent_index,
            chunk_type=ChunkType.SECTION,
            level=1,
            children_indices=[c.chunk_index for c in group],
            metadata=metadata,
        )
        parent_id = f"parent_{parent_index}"
        parent.parent_id = parent_id
        for child in group:
            child.parent_index = parent_index
            child.parent_id = parent_id
            child.children_indices = []
            # Legacy linkage key the RAG indexing consumers read.
            child.metadata["parent_chunk_index"] = parent_index
        parents.append(parent)
    return parents


def _derive_parent_child(
    text: str,
    flat: List[Dict[str, Any]],
    max_size: int,
    parent_size_multiplier: int,
) -> Tuple[List[StructuredChunk], List[StructuredChunk]]:
    """Derive (children, parents) from the engine's flat hierarchical output."""
    children = _build_children(flat)
    parents = _group_into_parents(
        text, children, max_size, parent_size_multiplier
    )
    return children, parents


def chunk_with_parent_retrieval(
    text: str,
    max_size: Optional[int] = None,
    overlap: Optional[int] = None,
    parent_size_multiplier: int = DEFAULT_PARENT_SIZE_MULTIPLIER,
    **opts: Any,
) -> Dict[str, Any]:
    """Chunk text with parent document retrieval support (legacy shape).

    Calls the engine's hierarchical ``structure_aware`` path and derives the
    parent/child structure from its section ancestry, preserving the legacy
    ``EnhancedChunkingService.chunk_with_parent_retrieval`` return shape:
    ``chunks`` (retrieval chunks with ``metadata.parent_chunk_index`` plus a
    ``parent_id`` reference), ``parent_chunks`` (parents in the legacy
    ``to_dict`` shape with ``children`` references) and ``metadata`` with the
    legacy count keys.

    Args:
        text: Text to chunk.
        max_size: Element budget per retrieval chunk (engine semantics for
            ``structure_aware`` grouping). The legacy ``chunk_size`` kwarg is
            accepted as an alias.
        overlap: Element overlap between retrieval chunks. The legacy
            ``chunk_overlap`` kwarg is accepted as an alias.
        parent_size_multiplier: A parent holds at most this many times
            ``max_size`` grouped elements.
        **opts: Legacy kwargs (e.g. ``chunk_size``/``chunk_overlap``) and
            ignored legacy flags, accepted for signature compatibility.

    Returns:
        Dictionary with ``chunks``, ``parent_chunks`` and ``metadata``.
    """
    max_size, overlap = _resolve_sizes(max_size, overlap, opts)
    clean_text, flat = _engine_flat(text, max_size, overlap)
    children, parents = _derive_parent_child(
        clean_text, flat, max_size, parent_size_multiplier
    )

    chunks: List[Dict[str, Any]] = []
    for child in children:
        chunk_dict = child.to_dict()
        chunk_dict["parent_id"] = child.parent_id
        chunks.append(chunk_dict)

    parent_chunks: List[Dict[str, Any]] = []
    for parent in parents:
        parent_dict = parent.to_dict()
        parent_dict["parent_id"] = parent.parent_id
        parent_dict["children"] = list(parent.children_indices)
        parent_chunks.append(parent_dict)

    try:
        parent_chunk_size = int(max_size) * int(parent_size_multiplier)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        parent_chunk_size = max_size

    logger.debug(
        "chunk_with_parent_retrieval: %d chunks, %d parent chunks "
        "(max_size=%d, overlap=%d)",
        len(chunks),
        len(parent_chunks),
        max_size,
        overlap,
    )
    return {
        "chunks": chunks,
        "parent_chunks": parent_chunks,
        "metadata": {
            "chunk_size": max_size,
            "parent_chunk_size": parent_chunk_size,
            "total_chunks": len(chunks),
            "total_parent_chunks": len(parent_chunks),
        },
    }


def chunk_text_with_structure(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    method: str = "hierarchical",
    parent_size_multiplier: int = DEFAULT_PARENT_SIZE_MULTIPLIER,
    preserve_structure: bool = True,
    clean_artifacts: bool = True,
    serialize_tables: bool = True,
) -> List[StructuredChunk]:
    """Structure-aware chunking returning legacy ``StructuredChunk`` objects.

    Args:
        content: Text to chunk.
        chunk_size: Element budget per chunk (engine ``structure_aware``
            grouping semantics).
        chunk_overlap: Element overlap between chunks.
        method: Legacy method name (``"hierarchical"``, ``"structural"``,
            ``"contextual"``). Accepted for signature compatibility; the
            engine's ``structure_aware`` strategy is the only
            structure-aware implementation (Q5 ruling).
        parent_size_multiplier: Used only to derive ``parent_index`` links
            on the returned chunks.
        preserve_structure: Ignored (structure is always preserved by the
            engine's hierarchical path).
        clean_artifacts: Ignored (the engine sanitizes input internally).
        serialize_tables: Ignored (the deleted bespoke table serialization
            went with ``DocumentStructureParser``).

    Returns:
        List of StructuredChunk objects in document order.
    """
    clean_text, flat = _engine_flat(content, chunk_size, chunk_overlap)
    children, _parents = _derive_parent_child(
        clean_text, flat, chunk_size, parent_size_multiplier
    )
    return children
