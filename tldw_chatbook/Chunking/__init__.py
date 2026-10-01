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


_LAZY_EXPORTS = {
    "Chunker": (".Chunk_Lib", "Chunker"),
    "ChunkingError": (".Chunk_Lib", "ChunkingError"),
    "InvalidChunkingMethodError": (".Chunk_Lib", "InvalidChunkingMethodError"),
    "InvalidInputError": (".Chunk_Lib", "InvalidInputError"),
    "LanguageDetectionError": (".Chunk_Lib", "LanguageDetectionError"),
    "improved_chunking_process": (".Chunk_Lib", "improved_chunking_process"),
    "chunk_for_embedding": (".Chunk_Lib", "chunk_for_embedding"),
    "process_document_with_metadata": (".Chunk_Lib", "process_document_with_metadata"),
    "DEFAULT_CHUNK_OPTIONS": (".Chunk_Lib", "DEFAULT_CHUNK_OPTIONS"),
    "ENGINE_VERSION": (".Chunk_Lib", "ENGINE_VERSION"),
    "LanguageChunkerFactory": (".language_chunkers", "LanguageChunkerFactory"),
    "ChineseChunker": (".language_chunkers", "ChineseChunker"),
    "JapaneseChunker": (".language_chunkers", "JapaneseChunker"),
    "DefaultChunker": (".language_chunkers", "DefaultChunker"),
    "TokenBasedChunker": (".token_chunker", "TokenBasedChunker"),
    "create_token_chunker": (".token_chunker", "create_token_chunker"),
    "Chunk_Lib": (".Chunk_Lib", None),
    "language_chunkers": (".language_chunkers", None),
    "token_chunker": (".token_chunker", None),
    "engine": (".engine", None),
}


def __getattr__(name: str):
    """Resolve the original dependency object only when a caller uses it."""
    from importlib import import_module

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    owner = import_module(module, __name__)
    value = getattr(owner, attribute) if attribute else owner
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
