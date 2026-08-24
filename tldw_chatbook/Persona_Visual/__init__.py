"""Separate local Persona Visual operational-state contracts.

PEP-562 lazy facade (TASK-21103): this package init used to import
``authoring``/``importer`` (and through them ``assets``, ``publication``,
``repository`` -- four modules with module-level ``from PIL import Image``)
eagerly, so importing ANY ``Persona_Visual`` submodule executed nearly the
whole 6,633-LOC tree and put PIL on the app boot path via the
Persona Buddy controller chain. Every public name still resolves at
``tldw_chatbook.Persona_Visual.<name>``, but the defining submodule is only
executed on first attribute access. Guarded by
``Tests/Packaging/test_persona_buddy_import_closure.py``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .authoring import (
        PersonaVisualAuthoringDraft,
        PersonaVisualAuthoringError,
        PersonaVisualDraftAsset,
        PersonaVisualDraftInventory,
        PersonaVisualDraftRow,
        add_persona_visual_custom_state,
        clear_persona_visual_draft_state,
        create_persona_visual_draft,
        inspect_persona_visual_draft,
        persona_visual_draft_from_graph,
        persona_visual_draft_publication_snapshot,
        replace_persona_visual_draft_state,
    )
    from .contracts import (
        ALLOWED_ASSET_EXTENSIONS,
        ALLOWED_ASSET_MIME_TYPES,
        ALLOWED_STATE_CATALOG_KINDS,
        ALLOWED_TRIGGER_SOURCES,
        MAX_ASSET_COUNT,
        MAX_ASSET_DIMENSION,
        MAX_ASSET_TOTAL_BYTES,
        MAX_CUSTOM_STATES,
        MAX_FALLBACK_DEPTH,
        MAX_FRAMES_PER_ANIMATION,
        MAX_TRIGGERS,
        REQUIRED_STATES,
        RESERVED_STATES,
        PersonaVisualAlignment,
        PersonaVisualAnimation,
        PersonaVisualCapability,
        PersonaVisualCatalogEntry,
        PersonaVisualFrame,
        PersonaVisualManifest,
        PersonaVisualManifestError,
        PersonaVisualRegion,
        PersonaVisualStateSelection,
        PersonaVisualStaticSelection,
        PersonaVisualTrigger,
        inspect_persona_visual_capability,
        resolve_manifest_state,
    )
    from .importer import (
        PERSONA_VISUAL_PACK_SCHEMA,
        PersonaVisualImportError,
        PersonaVisualImportReview,
        cleanup_persona_visual_import_review,
        import_persona_visual_pack,
        persona_visual_import_source_root,
    )
    from .validation import validate_persona_visual_manifest

_EXPORTS = {
    "PersonaVisualAuthoringDraft": "authoring",
    "PersonaVisualAuthoringError": "authoring",
    "PersonaVisualDraftAsset": "authoring",
    "PersonaVisualDraftInventory": "authoring",
    "PersonaVisualDraftRow": "authoring",
    "add_persona_visual_custom_state": "authoring",
    "clear_persona_visual_draft_state": "authoring",
    "create_persona_visual_draft": "authoring",
    "inspect_persona_visual_draft": "authoring",
    "persona_visual_draft_from_graph": "authoring",
    "persona_visual_draft_publication_snapshot": "authoring",
    "replace_persona_visual_draft_state": "authoring",
    "ALLOWED_ASSET_EXTENSIONS": "contracts",
    "ALLOWED_ASSET_MIME_TYPES": "contracts",
    "ALLOWED_STATE_CATALOG_KINDS": "contracts",
    "ALLOWED_TRIGGER_SOURCES": "contracts",
    "MAX_ASSET_COUNT": "contracts",
    "MAX_ASSET_DIMENSION": "contracts",
    "MAX_ASSET_TOTAL_BYTES": "contracts",
    "MAX_CUSTOM_STATES": "contracts",
    "MAX_FALLBACK_DEPTH": "contracts",
    "MAX_FRAMES_PER_ANIMATION": "contracts",
    "MAX_TRIGGERS": "contracts",
    "REQUIRED_STATES": "contracts",
    "RESERVED_STATES": "contracts",
    "PersonaVisualAlignment": "contracts",
    "PersonaVisualAnimation": "contracts",
    "PersonaVisualCapability": "contracts",
    "PersonaVisualCatalogEntry": "contracts",
    "PersonaVisualFrame": "contracts",
    "PersonaVisualManifest": "contracts",
    "PersonaVisualManifestError": "contracts",
    "PersonaVisualRegion": "contracts",
    "PersonaVisualStateSelection": "contracts",
    "PersonaVisualStaticSelection": "contracts",
    "PersonaVisualTrigger": "contracts",
    "inspect_persona_visual_capability": "contracts",
    "resolve_manifest_state": "contracts",
    "PERSONA_VISUAL_PACK_SCHEMA": "importer",
    "PersonaVisualImportError": "importer",
    "PersonaVisualImportReview": "importer",
    "cleanup_persona_visual_import_review": "importer",
    "import_persona_visual_pack": "importer",
    "persona_visual_import_source_root": "importer",
    "validate_persona_visual_manifest": "validation",
}

__all__ = [
    "ALLOWED_ASSET_EXTENSIONS",
    "ALLOWED_ASSET_MIME_TYPES",
    "ALLOWED_STATE_CATALOG_KINDS",
    "ALLOWED_TRIGGER_SOURCES",
    "MAX_ASSET_COUNT",
    "MAX_ASSET_DIMENSION",
    "MAX_ASSET_TOTAL_BYTES",
    "MAX_CUSTOM_STATES",
    "MAX_FALLBACK_DEPTH",
    "MAX_FRAMES_PER_ANIMATION",
    "MAX_TRIGGERS",
    "REQUIRED_STATES",
    "RESERVED_STATES",
    "PersonaVisualAuthoringDraft",
    "PersonaVisualAuthoringError",
    "PersonaVisualDraftAsset",
    "PersonaVisualDraftInventory",
    "PersonaVisualDraftRow",
    "PersonaVisualAlignment",
    "PersonaVisualAnimation",
    "PersonaVisualCapability",
    "PersonaVisualCatalogEntry",
    "PersonaVisualFrame",
    "PersonaVisualImportError",
    "PersonaVisualImportReview",
    "PersonaVisualManifest",
    "PersonaVisualManifestError",
    "PersonaVisualRegion",
    "PersonaVisualStateSelection",
    "PersonaVisualStaticSelection",
    "PersonaVisualTrigger",
    "PERSONA_VISUAL_PACK_SCHEMA",
    "inspect_persona_visual_capability",
    "add_persona_visual_custom_state",
    "clear_persona_visual_draft_state",
    "cleanup_persona_visual_import_review",
    "create_persona_visual_draft",
    "inspect_persona_visual_draft",
    "import_persona_visual_pack",
    "persona_visual_draft_from_graph",
    "persona_visual_draft_publication_snapshot",
    "persona_visual_import_source_root",
    "replace_persona_visual_draft_state",
    "resolve_manifest_state",
    "validate_persona_visual_manifest",
]


def __getattr__(name: str) -> object:
    """Resolve a public export by importing its defining submodule on demand.

    Args:
        name: The requested package attribute.

    Returns:
        The submodule attribute the facade re-exports.

    Raises:
        AttributeError: If ``name`` is not a public Persona_Visual export.
    """
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{submodule}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the package's public exports for introspection."""
    return sorted(set(globals()) | set(__all__))
