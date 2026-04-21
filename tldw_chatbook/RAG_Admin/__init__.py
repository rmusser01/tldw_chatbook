"""Shared local/server retrieval-admin seam for chunking templates and collections."""

from .local_rag_admin_service import LocalRAGAdminService
from .rag_admin_normalizers import normalize_collection_record, normalize_template_record
from .rag_admin_scope_service import RAGAdminBackend, RAGAdminScopeService
from .server_rag_admin_service import ServerRAGAdminService

__all__ = [
    "LocalRAGAdminService",
    "normalize_collection_record",
    "normalize_template_record",
    "RAGAdminBackend",
    "RAGAdminScopeService",
    "ServerRAGAdminService",
]
