"""Translation remote parity services."""

from .server_translation_service import ServerTranslationService
from .translation_scope_service import TranslationBackend, TranslationScopeService

__all__ = [
    "ServerTranslationService",
    "TranslationBackend",
    "TranslationScopeService",
]
