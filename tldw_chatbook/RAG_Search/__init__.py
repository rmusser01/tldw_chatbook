# __init__.py
# Description: RAG Search package initialization
"""Lazy facade over the RAG Search package (PEP 562, task-21102).

The new simplified RAG implementation is in the ``simplified`` subdirectory;
for backward compatibility the main components are re-exported here. The
re-exports are resolved on FIRST ATTRIBUTE ACCESS rather than at package
import: this ``__init__`` used to eagerly import ``.simplified`` (its whole
service tree) and ``.chunking_service`` (the full ``Chunking`` shim +
vendored engine, ~15k LOC), which meant that importing ANY ``RAG_Search``
submodule -- e.g. the lightweight ``ingestion_indexing`` seam that
``Local_Ingestion.local_file_ingestion`` needs at boot -- executed all of it.
Guarded by ``Tests/Packaging/test_chunking_import_closure.py``.

Semantics preserved from the eager version: when the underlying import fails
(missing optional RAG dependencies), the resolved name degrades to a stub
class whose constructor raises ``ImportError`` -- the failure just surfaces
at first use instead of at package import.
"""

from typing import Any

#: Re-exported name -> (submodule, attribute) it resolves from.
#: ``IndexingService`` is a backward-compatibility alias for ``RAGService``.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EmbeddingsService": ("simplified", "EmbeddingsService"),
    "RAGService": ("simplified", "RAGService"),
    "IndexingService": ("simplified", "RAGService"),
    "RAGConfig": ("simplified", "RAGConfig"),
    "SearchResult": ("simplified", "SearchResult"),
    "SearchResultWithCitations": ("simplified", "SearchResultWithCitations"),
    "create_rag_service": ("simplified", "create_rag_service"),
    "create_config_for_collection": ("simplified", "create_config_for_collection"),
    "create_config_for_testing": ("simplified", "create_config_for_testing"),
    "ChunkingService": ("chunking_service", "ChunkingService"),
}

__all__ = [
    "EmbeddingsService",
    "ChunkingService",
    "IndexingService",
    "RAGService",
    "RAGConfig",
    "SearchResult",
    "SearchResultWithCitations",
    "create_rag_service",
    "create_config_for_collection",
    "create_config_for_testing",
]


def _unavailable_stub(name: str, error: ImportError) -> type:
    """Build the stub class the eager fallback used to define at import time.

    Args:
        name: The re-exported name being resolved.
        error: The ImportError that made the real implementation unavailable.

    Returns:
        A class whose constructor raises ``ImportError``, matching the
        legacy stub behavior for missing RAG dependencies.
    """
    message = (
        "RAG services not available. Please check dependencies. "
        f"({name} unavailable: {error})"
    )

    class _UnavailableRAGComponent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(message)

    _UnavailableRAGComponent.__name__ = name
    _UnavailableRAGComponent.__qualname__ = name
    return _UnavailableRAGComponent


def __getattr__(name: str) -> Any:
    """Resolve a lazy re-export on first access (PEP 562)."""
    try:
        submodule, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    try:
        from importlib import import_module

        value = getattr(import_module(f".{submodule}", __name__), attribute)
    except ImportError as e:
        import logging

        logging.getLogger(__name__).error(
            f"Failed to import simplified RAG services: {e}"
        )
        value = _unavailable_stub(name, e)

    globals()[name] = value  # cache: subsequent accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
