"""Library destination state and service contracts."""

from importlib import import_module
from typing import TYPE_CHECKING

from .library_collections_service import (
    LibraryCollectionRecord,
    LibraryCollectionsService,
    LocalLibraryCollectionsService,
)
from .library_tool_contract import (
    LIBRARY_TOOL_DESCRIPTORS,
    LibraryToolDescriptor,
    LibraryToolError,
)
from .local_library_tool_service import LocalLibraryToolService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .collections_capture_repository import CollectionsCaptureRepository
    from .collections_capture_service import (
        CollectionsCaptureScopeService,
        LocalCollectionsCaptureService,
        build_local_capture_authority,
        build_server_capture_authority,
    )
    from .collections_legacy_recovery import LegacyCollectionsRecovery
    from .collections_offline_store import CollectionsOfflineStore
    from .server_collections_capture_service import ServerCollectionsCaptureService


_LAZY_CAPTURE_EXPORTS = {
    "CollectionsCaptureRepository": ".collections_capture_repository",
    "CollectionsCaptureScopeService": ".collections_capture_service",
    "CollectionsOfflineStore": ".collections_offline_store",
    "LegacyCollectionsRecovery": ".collections_legacy_recovery",
    "LocalCollectionsCaptureService": ".collections_capture_service",
    "ServerCollectionsCaptureService": ".server_collections_capture_service",
    "build_local_capture_authority": ".collections_capture_service",
    "build_server_capture_authority": ".collections_capture_service",
}


def __getattr__(name: str) -> object:
    """Resolve capture implementation exports only when a caller needs them."""
    module_name = _LAZY_CAPTURE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List eager and deferred Library exports."""
    return sorted(set(globals()) | _LAZY_CAPTURE_EXPORTS)

__all__ = [
    "LIBRARY_TOOL_DESCRIPTORS",
    "CollectionsCaptureRepository",
    "CollectionsCaptureScopeService",
    "CollectionsOfflineStore",
    "LegacyCollectionsRecovery",
    "LibraryCollectionRecord",
    "LibraryCollectionsService",
    "LibraryToolDescriptor",
    "LibraryToolError",
    "LocalLibraryCollectionsService",
    "LocalCollectionsCaptureService",
    "ServerCollectionsCaptureService",
    "LocalLibraryToolService",
    "build_local_capture_authority",
    "build_server_capture_authority",
]
