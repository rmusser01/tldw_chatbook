"""Server/local output artifacts and templates interoperability services."""

from .outputs_scope_service import OutputsBackend, OutputsScopeService
from .server_outputs_service import ServerOutputsService

__all__ = ["OutputsBackend", "OutputsScopeService", "ServerOutputsService"]
