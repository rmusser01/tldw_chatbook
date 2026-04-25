"""Remote web-clipper interoperability services."""

from .server_web_clipper_service import ServerWebClipperService
from .web_clipper_scope_service import WebClipperBackend, WebClipperScopeService

__all__ = ["ServerWebClipperService", "WebClipperBackend", "WebClipperScopeService"]
