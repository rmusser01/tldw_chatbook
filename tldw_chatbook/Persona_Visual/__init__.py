"""Exact lazy public exports; recovery declarations do not bootstrap runtime."""

from importlib import import_module

_EXPORTS = {
    "PersonaVisualAuthoringDraft": (".authoring", "PersonaVisualAuthoringDraft"),
    "PersonaVisualAuthoringError": (".authoring", "PersonaVisualAuthoringError"),
    "PersonaVisualDraftAsset": (".authoring", "PersonaVisualDraftAsset"),
    "PersonaVisualDraftInventory": (".authoring", "PersonaVisualDraftInventory"),
    "PersonaVisualDraftRow": (".authoring", "PersonaVisualDraftRow"),
    "add_persona_visual_custom_state": (
        ".authoring",
        "add_persona_visual_custom_state",
    ),
    "clear_persona_visual_draft_state": (
        ".authoring",
        "clear_persona_visual_draft_state",
    ),
    "create_persona_visual_draft": (".authoring", "create_persona_visual_draft"),
    "inspect_persona_visual_draft": (".authoring", "inspect_persona_visual_draft"),
    "persona_visual_draft_from_graph": (
        ".authoring",
        "persona_visual_draft_from_graph",
    ),
    "persona_visual_draft_publication_snapshot": (
        ".authoring",
        "persona_visual_draft_publication_snapshot",
    ),
    "replace_persona_visual_draft_state": (
        ".authoring",
        "replace_persona_visual_draft_state",
    ),
    "ALLOWED_ASSET_EXTENSIONS": (".contracts", "ALLOWED_ASSET_EXTENSIONS"),
    "ALLOWED_ASSET_MIME_TYPES": (".contracts", "ALLOWED_ASSET_MIME_TYPES"),
    "ALLOWED_STATE_CATALOG_KINDS": (".contracts", "ALLOWED_STATE_CATALOG_KINDS"),
    "ALLOWED_TRIGGER_SOURCES": (".contracts", "ALLOWED_TRIGGER_SOURCES"),
    "MAX_ASSET_COUNT": (".contracts", "MAX_ASSET_COUNT"),
    "MAX_ASSET_DIMENSION": (".contracts", "MAX_ASSET_DIMENSION"),
    "MAX_ASSET_TOTAL_BYTES": (".contracts", "MAX_ASSET_TOTAL_BYTES"),
    "MAX_CUSTOM_STATES": (".contracts", "MAX_CUSTOM_STATES"),
    "MAX_FALLBACK_DEPTH": (".contracts", "MAX_FALLBACK_DEPTH"),
    "MAX_FRAMES_PER_ANIMATION": (".contracts", "MAX_FRAMES_PER_ANIMATION"),
    "MAX_TRIGGERS": (".contracts", "MAX_TRIGGERS"),
    "REQUIRED_STATES": (".contracts", "REQUIRED_STATES"),
    "RESERVED_STATES": (".contracts", "RESERVED_STATES"),
    "PersonaVisualAlignment": (".contracts", "PersonaVisualAlignment"),
    "PersonaVisualAnimation": (".contracts", "PersonaVisualAnimation"),
    "PersonaVisualCapability": (".contracts", "PersonaVisualCapability"),
    "PersonaVisualCatalogEntry": (".contracts", "PersonaVisualCatalogEntry"),
    "PersonaVisualFrame": (".contracts", "PersonaVisualFrame"),
    "PersonaVisualManifest": (".contracts", "PersonaVisualManifest"),
    "PersonaVisualManifestError": (".contracts", "PersonaVisualManifestError"),
    "PersonaVisualRegion": (".contracts", "PersonaVisualRegion"),
    "PersonaVisualStateSelection": (".contracts", "PersonaVisualStateSelection"),
    "PersonaVisualStaticSelection": (".contracts", "PersonaVisualStaticSelection"),
    "PersonaVisualTrigger": (".contracts", "PersonaVisualTrigger"),
    "inspect_persona_visual_capability": (
        ".contracts",
        "inspect_persona_visual_capability",
    ),
    "resolve_manifest_state": (".contracts", "resolve_manifest_state"),
    "PERSONA_VISUAL_PACK_SCHEMA": (".importer", "PERSONA_VISUAL_PACK_SCHEMA"),
    "PersonaVisualImportError": (".importer", "PersonaVisualImportError"),
    "PersonaVisualImportReview": (".importer", "PersonaVisualImportReview"),
    "cleanup_persona_visual_import_review": (
        ".importer",
        "cleanup_persona_visual_import_review",
    ),
    "import_persona_visual_pack": (".importer", "import_persona_visual_pack"),
    "persona_visual_import_source_root": (
        ".importer",
        "persona_visual_import_source_root",
    ),
    "validate_persona_visual_manifest": (
        ".validation",
        "validate_persona_visual_manifest",
    ),
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


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value
