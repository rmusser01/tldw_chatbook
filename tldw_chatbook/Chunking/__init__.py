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

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "Chunker": ".Chunk_Lib",
    "ChunkingError": ".Chunk_Lib",
    "InvalidChunkingMethodError": ".Chunk_Lib",
    "InvalidInputError": ".Chunk_Lib",
    "LanguageDetectionError": ".Chunk_Lib",
    "improved_chunking_process": ".Chunk_Lib",
    "chunk_for_embedding": ".Chunk_Lib",
    "process_document_with_metadata": ".Chunk_Lib",
    "DEFAULT_CHUNK_OPTIONS": ".Chunk_Lib",
    "ENGINE_VERSION": ".Chunk_Lib",
    "LanguageChunkerFactory": ".language_chunkers",
    "ChineseChunker": ".language_chunkers",
    "JapaneseChunker": ".language_chunkers",
    "DefaultChunker": ".language_chunkers",
    "TokenBasedChunker": ".token_chunker",
    "create_token_chunker": ".token_chunker",
}

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


def __getattr__(name: str) -> Any:
    # Pure migration helpers must not load runtime configuration on import.
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORT_MODULES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
