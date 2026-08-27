"""Research Workspace authority adapters and normalized contracts.

Lazy re-exports (PEP 562). The names in ``__all__`` resolve on first
attribute access via module ``__getattr__`` -- the same pattern as
``tldw_api/__init__.py`` and ``Local_Ingestion/__init__.py``. This package
sits on the boot import path through its cheapest member:
``Library/library_ingest_jobs.py`` (an ``app.py`` module-scope dependency)
imports ``source_operations.validate_source_operation_id``, and ``app.py``
itself imports four more stdlib-light submodules directly. When this
``__init__`` eagerly re-exported the whole tree, that one validator import
also executed ``server_adapter`` -> ``tldw_api.notes_workspace_schemas``
(26 pydantic models) plus the controller/overlay/layout modules only the
Research Workspace screen needs -- ~48 ms and 8 boot modules for a 6 ms
validator (TASK-23023; same class as the TASK-21102/21107 facade leaks).

Regular callers (``from tldw_chatbook.Research_Workspace import
ResearchWorkspaceController``) are unaffected: the first access imports the
owning submodule, returns the identical object, and caches it on this
module so later lookups are plain attribute access. Direct submodule
imports still run this ``__init__`` first per standard import semantics,
but that is now free of eager submodule imports of its own.

Guarded by ``Tests/Packaging/test_research_workspace_import_closure.py``.
"""

from typing import Any

__all__ = [
    "BoundedPageResult",
    "CapabilityUnavailableError",
    "LocalResearchWorkspaceAdapter",
    "ProcessingRoute",
    "QualifiedWorkspaceRef",
    "ResearchCatalogItem",
    "ResearchCapability",
    "ResearchPanePreferences",
    "ResearchNoteConflictError",
    "ResearchNotePage",
    "ResearchNotePageRequest",
    "ResearchNoteSaveRequest",
    "ResearchQuickNote",
    "ResearchQuickNotesService",
    "ResearchPresentationOverlayStore",
    "ResearchRequestContext",
    "ResearchSourcePreview",
    "ResearchSourcePage",
    "ResearchSourceSummary",
    "SourceSelectionResult",
    "ResearchSurfaceRequest",
    "ResearchWorkspaceController",
    "ResearchWorkspaceCatalogState",
    "ResearchWorkspacePort",
    "ResearchWorkspaceSummary",
    "ServerResearchWorkspaceAdapter",
    "RetrievalMode",
    "SourceReadiness",
    "SourceReadinessState",
    "SourceIdentityMismatchError",
    "WorkspaceDataSource",
]

#: Which submodule owns each re-exported name. Flat mapping so
#: ``__getattr__`` only ever imports the one submodule a caller asked for.
_SUBMODULE_BY_NAME = {
    "BoundedPageResult": "contracts",
    "CapabilityUnavailableError": "contracts",
    "ProcessingRoute": "contracts",
    "QualifiedWorkspaceRef": "contracts",
    "ResearchCatalogItem": "contracts",
    "ResearchCapability": "contracts",
    "ResearchSourcePreview": "contracts",
    "ResearchSourcePage": "contracts",
    "ResearchSourceSummary": "contracts",
    "SourceSelectionResult": "contracts",
    "ResearchWorkspacePort": "contracts",
    "ResearchWorkspaceSummary": "contracts",
    "RetrievalMode": "contracts",
    "SourceReadiness": "contracts",
    "SourceReadinessState": "contracts",
    "SourceIdentityMismatchError": "contracts",
    "WorkspaceDataSource": "contracts",
    "ResearchRequestContext": "controller",
    "ResearchSurfaceRequest": "controller",
    "ResearchWorkspaceCatalogState": "controller",
    "ResearchWorkspaceController": "controller",
    "ResearchPanePreferences": "layout_state",
    "ResearchPresentationOverlayStore": "overlay_store",
    "LocalResearchWorkspaceAdapter": "local_adapter",
    "ServerResearchWorkspaceAdapter": "server_adapter",
    "ResearchNoteConflictError": "quick_notes",
    "ResearchNotePage": "quick_notes",
    "ResearchNotePageRequest": "quick_notes",
    "ResearchNoteSaveRequest": "quick_notes",
    "ResearchQuickNote": "quick_notes",
    "ResearchQuickNotesService": "quick_notes",
}


def __getattr__(name: str) -> Any:
    submodule_name = _SUBMODULE_BY_NAME.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    submodule = importlib.import_module(f".{submodule_name}", __name__)
    value = getattr(submodule, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> "list[str]":
    return sorted(set(__all__) | set(globals()))
