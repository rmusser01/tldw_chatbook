# __init__.py
"""
Chunking module for flexible text chunking.

Templates are DB rows: name resolution lives in
``template_runtime.resolve_template`` at the service layer (spec §8.2), and
``Chunker``/``improved_chunking_process`` accept only pre-resolved template
dicts. The former file store (``Chunking/templates/``) and its manager
module (``chunking_templates.py``, re-exported here until its deletion) are
gone -- a breaking change to this package's namespace recorded in the
CHANGELOG. The vendored engine's ``ChunkingTemplate`` is deliberately NOT
re-exported: nothing outside the service layer resolves templates.
"""

from .Chunk_Lib import (
    Chunker,
    ChunkingError,
    InvalidChunkingMethodError,
    InvalidInputError,
    LanguageDetectionError,
    improved_chunking_process,
    chunk_for_embedding,
    process_document_with_metadata,
    DEFAULT_CHUNK_OPTIONS,
    ENGINE_VERSION,
)

from .language_chunkers import (
    LanguageChunkerFactory,
    ChineseChunker,
    JapaneseChunker,
    DefaultChunker,
)

from .token_chunker import TokenBasedChunker, create_token_chunker

__all__ = [
    # Main chunking classes
    "Chunker",
    "improved_chunking_process",
    "chunk_for_embedding",
    "process_document_with_metadata",
    "DEFAULT_CHUNK_OPTIONS",
    "ENGINE_VERSION",
    # Language support
    "LanguageChunkerFactory",
    "ChineseChunker",
    "JapaneseChunker",
    "DefaultChunker",
    # Token support
    "TokenBasedChunker",
    "create_token_chunker",
    # Exceptions
    "ChunkingError",
    "InvalidChunkingMethodError",
    "InvalidInputError",
    "LanguageDetectionError",
]
