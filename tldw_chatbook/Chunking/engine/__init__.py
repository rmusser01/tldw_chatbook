"""Chatbook's Chunking engine package init — NOT vendored (spec §5.1).

Re-exports the phase-1 public surface only. Upstream's own __init__ pulls in
#2/#3/#6-deferred modules and would coexist badly with the Chunk_Lib shim.
"""
from .base import ChunkerConfig, ChunkingMethod
from .chunker import Chunker
from .exceptions import (
    ChunkingError, InvalidInputError, InvalidChunkingMethodError, TokenizerError,
    TemplateError, LanguageNotSupportedError, ChunkSizeError, ProcessingError,
    ConfigurationError, CacheError,
)

__all__ = [
    "Chunker", "ChunkerConfig", "ChunkingMethod",
    "ChunkingError", "InvalidInputError", "InvalidChunkingMethodError",
    "TokenizerError", "TemplateError", "LanguageNotSupportedError",
    "ChunkSizeError", "ProcessingError", "ConfigurationError", "CacheError",
]
