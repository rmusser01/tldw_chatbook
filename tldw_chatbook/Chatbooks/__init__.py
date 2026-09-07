# Chatbooks module - Knowledge pack creation and management
"""
Chatbooks Module
----------------

This module provides functionality for creating, managing, and sharing
knowledge packs (chatbooks) that contain curated content from multiple
databases in the tldw_chatbook application.

Main components:
- chatbook_creator.py: Package creation logic
- chatbook_importer.py: Import and validation
- chatbook_models.py: Data structures and schemas
- conflict_resolver.py: Handle duplicate content during import
- error_handler.py: Comprehensive error handling
"""

from importlib import import_module

__all__ = [
    "ChatbookCreator",
    "ChatbookImporter",
    "Chatbook",
    "ChatbookManifest",
    "ChatbookContent",
    "LocalChatbookService",
    "ServerChatbookService",
    "ChatbookError",
    "ChatbookErrorHandler",
    "ChatbookErrorType",
]

_LAZY_EXPORTS = {
    "ChatbookCreator": (".chatbook_creator", "ChatbookCreator"),
    "ChatbookImporter": (".chatbook_importer", "ChatbookImporter"),
    "Chatbook": (".chatbook_models", "Chatbook"),
    "ChatbookManifest": (".chatbook_models", "ChatbookManifest"),
    "ChatbookContent": (".chatbook_models", "ChatbookContent"),
    "LocalChatbookService": (".local_chatbook_service", "LocalChatbookService"),
    "ServerChatbookService": (".server_chatbook_service", "ServerChatbookService"),
    "ChatbookError": (".error_handler", "ChatbookError"),
    "ChatbookErrorHandler": (".error_handler", "ChatbookErrorHandler"),
    "ChatbookErrorType": (".error_handler", "ChatbookErrorType"),
}


def __getattr__(name: str):
    """Resolve the existing public API without importing unused engines."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
