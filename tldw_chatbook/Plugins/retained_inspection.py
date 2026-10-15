"""Closed historical interpretation selection from authenticated retained records."""

from collections.abc import Mapping, Sequence
from typing import Any

from .inspection import _inspect_capture, catalog_overlay_from
from .models import AcquisitionReference, CatalogBinding, PackageInspection
from .package_files import PackageCapture, canonical_json, parse_document

_RECIPES = {
    ("portable", "chatbook-portable/1"): ("dev_20261001", "branch_7b4d5c"),
    ("openai", "chatbook-openai/2026-10-01.1"): ("dev_20261001",),
    ("cursor", "chatbook-cursor/2026-10-01.1"): ("dev_20261001",),
    ("openai", "chatbook-codex/1"): ("branch_7b4d5c",),
    ("cursor", "chatbook-cursor/1"): ("branch_7b4d5c",),
    ("portable", "chatbook-portable/2"): ("integrated_2",),
    ("openai", "chatbook-codex/2"): ("integrated_2",),
    ("cursor", "chatbook-cursor/2"): ("integrated_2",),
}
_REVISION_FIELDS = (
    "dialect", "format_version", "adapter_version", "root_manifest",
    "overlay_identities", "content_digest", "activation_blockers", "rejected",
)
_PROVENANCE = ("source_identity", "source_digest", "materialized_identity", "link_targets")

_UNSET_CATALOG = object()


def _source_bindings(revision: Mapping[str, Any]) -> dict[str, Any]:
    """Parse only the two closed provenance forms, without granting authority."""
    bindings = {}
    for name, model in (("catalog_binding", CatalogBinding),
                        ("acquisition_reference", AcquisitionReference)):
        if revision.get(name) is not None:
            bindings[name] = model.model_validate(revision[name])
    if len(bindings) > 1:
        raise ValueError("ambiguous_acquisition_binding")
    return bindings


def _verify_retained_source(revision: Mapping[str, Any], coordinator) -> None:
    """Require the actual signed producer/physical/bytes evidence before selection."""
    bindings = _source_bindings(revision)
    if not bindings:
        return
    if coordinator is None:
        raise ValueError("catalog owner required")
    from .acquisition import verify_reference

    catalog = bindings.get("catalog_binding")
    references = ((catalog.catalog, catalog.package) if catalog is not None
                  else (bindings["acquisition_reference"],))
    for reference in references:
        verify_reference(coordinator.owner, coordinator.authority, reference)


def _retained_candidates(
    capture: PackageCapture, revision: Mapping[str, Any],
    *, catalog_overlay: Any = _UNSET_CATALOG,
) -> tuple[PackageInspection, ...]:
    """Try only interpretations issued under this exact stored identity."""
    recipes = _RECIPES.get((revision.get("dialect"), revision.get("adapter_version")))
    if recipes is None:
        raise ValueError("retained_adapter_unsupported")
    bindings = _source_bindings(revision)
    candidates = []
    for recipe in recipes:
        try:
            overlay = catalog_overlay
            if overlay is _UNSET_CATALOG:
                overlay = (None if recipe == "dev_20261001"
                           else catalog_overlay_from(revision["overlay_identities"]))
            candidate = _inspect_capture(
                capture, dialect=revision["dialect"], catalog_overlay=overlay, recipe=recipe
            )
            if bindings:
                from .source_service import bind_source_inspection

                candidate = bind_source_inspection(candidate, **bindings)
            candidates.append(candidate)
        except (TypeError, ValueError):
            # A malformed carrier never falls back to a fresh adapter identity.
            continue
    return tuple(candidates)


def _interpretation_key(inspection: PackageInspection) -> str:
    """Compare complete selected semantics, excluding only regenerated evidence."""
    data = inspection.model_dump(mode="json")
    for key in (*_PROVENANCE, "diagnostics", "candidates"):
        data.pop(key)
    for row in data["inventory"].values():
        row.pop("evidence")
        row.pop("availability")
    return canonical_json(data)


def _unique_interpretation(matches: Sequence[PackageInspection]) -> PackageInspection:
    unique = {}
    for inspection in matches:
        unique.setdefault(_interpretation_key(inspection), inspection)
    if not unique:
        raise ValueError("retained_interpretation_mismatch")
    if len(unique) != 1:
        raise ValueError("retained_interpretation_ambiguous")
    return next(iter(unique.values()))


def _matches_projection(revision, components, inspection) -> bool:
    from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
        canonical_json as authority_json, sha256_hex,
    )

    data = inspection.model_dump(mode="json")
    if any(data[key] != revision[key] for key in _REVISION_FIELDS):
        return False
    if (inspection.effective_digest != revision["revision_digest"]
        or sha256_hex(authority_json(parse_document(inspection.variables_json.encode())))
        != revision["variables_digest"]):
        return False
    expected = {row["component_id"]: row for row in components}
    if len(expected) != len(components) or set(expected) != set(inspection.inventory):
        return False
    for component_id, component in inspection.inventory.items():
        row = component.model_dump(mode="json")
        for key in ("availability", "evidence", "definition_json"):
            row.pop(key)
        row.update(
            installation_id=revision["installation_id"],
            revision_digest=revision["revision_digest"],
            definition_digest=sha256_hex(authority_json(parse_document(
                b'{"definition":' + component.definition_json.encode() + b'}'
            )["definition"])),
        )
        if row != expected[component_id]:
            return False
    return True


def _select_retained(
    revision: Mapping[str, Any], components: Sequence[Mapping[str, Any]],
    candidates: Sequence[PackageInspection],
) -> PackageInspection:
    """Match all authenticated fields before restoring acquisition provenance."""
    selected = _unique_interpretation([
        item for item in candidates if _matches_projection(revision, components, item)
    ])
    return selected.model_copy(update={key: revision[key] for key in _PROVENANCE})


def _reinspect_expected(
    capture: PackageCapture, expected: PackageInspection,
    *, catalog_overlay: Any = _UNSET_CATALOG,
) -> PackageInspection:
    """Bind an issued in-process review, including its original candidate menu."""
    from .review import inspection_identity

    if capture.errors:
        raise ValueError("stale review or unsafe package capture")
    if expected.materialized_identity is None and (
        str(capture.root) != expected.source_identity
        or capture.source_digest != expected.source_digest
        or dict(capture.link_targets) != dict(expected.link_targets)
    ):
        raise ValueError("stale review source provenance")
    matches = []
    for candidate in _retained_candidates(
        capture, expected.model_dump(mode="json"), catalog_overlay=catalog_overlay
    ):
        # Regular materialized files replace original links only after content matches.
        if candidate.content_digest != expected.content_digest:
            continue
        candidate = candidate.model_copy(update={key: getattr(expected, key) for key in _PROVENANCE})
        if inspection_identity(candidate) == inspection_identity(expected):
            matches.append(candidate)
    try:
        return _unique_interpretation(matches)
    except ValueError as exc:
        raise ValueError("stale review or changed retained package: " + str(exc)) from None
