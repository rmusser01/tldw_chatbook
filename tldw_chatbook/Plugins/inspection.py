"""Non-executing package inspection with deterministic interpretation selection."""

import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

from .adapters.portable import ComponentLimitError, inventory_package
from .models import Diagnostic, DialectCandidate, InspectionEvidence, PackageInspection
from .package_files import (
    PackageCapture,
    PackageFileError,
    canonical_json,
    capture_package,
    parse_document,
)
from .schemas import NAMESPACE, PLUGIN_SCHEMA, require, validate_manifest

MAX_COMPONENTS = 512
_MANIFEST_FIELDS = {
    "$schema",
    "name",
    "version",
    "description",
    "author",
    "homepage",
    "repository",
    "license",
    "keywords",
    "extensions",
}


CATALOG_PREFIX = "catalog-v1:"


def catalog_identity(overlay: dict) -> str:
    """Bound public catalog data inside the existing authenticated text carrier."""
    from tldw_chatbook.Agents.hooks_v2.validation import _json_tree

    value = _json_tree(overlay, max_bytes=16384)
    require(isinstance(value, dict), "catalog_overlay_invalid")

    from .adapters.codex import VENDOR_FIELDS

    require(set(value) <= VENDOR_FIELDS, "catalog_field_unsupported")
    # Public literals are not host credentials; declaration owners validate them.
    encoded = CATALOG_PREFIX + canonical_json(value)
    require(len(encoded.encode()) <= 16384, "catalog_overlay_limit")
    return encoded


def catalog_overlay_from(identities) -> dict | None:
    """Replay exactly one canonical reviewed carrier, never a live catalog."""
    require(
        isinstance(identities, (list, tuple))
        and all(isinstance(item, str) for item in identities),
        "catalog_overlay_invalid",
    )
    carriers = [item for item in identities if item.startswith("catalog-")]
    require(len(carriers) <= 1, "catalog_overlay_duplicate")
    if not carriers:
        return None
    carrier = carriers[0]
    require(carrier.startswith(CATALOG_PREFIX), "catalog_overlay_version")
    require(len(carrier.encode()) <= 16384, "catalog_overlay_limit")
    value = parse_document(carrier[len(CATALOG_PREFIX) :].encode())
    require(catalog_identity(value) == carrier, "catalog_overlay_noncanonical")
    return value


def inspect_package(
    root: Path, *, dialect: str | None = None, catalog_overlay: dict | None = None
) -> PackageInspection:
    """Inspect a bounded package without fetching, executing, or granting trust.

    Args:
        root: Existing canonical absolute package directory.
        dialect: Explicit portable, openai, or cursor interpretation.
        catalog_overlay: Public reviewed catalog data, never credentials.

    Returns:
        Immutable inventory and diagnostics. Rejected packages remain visible
        as failures; only a complete capture receives a content digest.
    """
    try:
        capture = capture_package(root)
    except PackageFileError as exc:
        return PackageInspection(
            source_identity=str(root),
            rejected=True,
            activation_blockers=(str(exc),),
            diagnostics=(Diagnostic(code=str(exc)),),
        )
    return inspect_capture(capture, dialect=dialect, catalog_overlay=catalog_overlay)


def inspect_capture(
    capture: PackageCapture,
    *,
    dialect: str | None = None,
    catalog_overlay: dict | None = None,
) -> PackageInspection:
    """Interpret exactly the captured bytes (also used by materialization)."""
    diagnostics = list(capture.diagnostics) + list(capture.errors)
    candidates = []
    manifest = None
    vendor_overlays = {}
    carrier = catalog_identity(catalog_overlay) if catalog_overlay is not None else None
    if "plugin.json" in capture.files:
        try:
            manifest = validate_manifest(capture.document("plugin.json"))
            for field in manifest.keys() - _MANIFEST_FIELDS:
                diagnostics.append(
                    Diagnostic(code="manifest_unknown_field", path="plugin.json")
                )
            extensions = manifest.get("extensions", {})
            overlays = (
                (f"plugin.json#/extensions/{NAMESPACE}",)
                if isinstance(extensions, dict) and NAMESPACE in extensions
                else ()
            )
            candidates.append(
                DialectCandidate(
                    dialect="portable",
                    format_version="1.0.0",
                    adapter_version="chatbook-portable/1",
                    root_manifest="plugin.json",
                    overlays=overlays,
                )
            )
        except PackageFileError as exc:
            diagnostics.append(Diagnostic(code=str(exc), path="plugin.json"))
            manifest = None
    elif (
        any(d.path == "plugin.json" for d in capture.errors)
        or "plugin.json" in capture.directories
    ):
        diagnostics.append(Diagnostic(code="manifest_unavailable", path="plugin.json"))

    # Candidate identity and the selected inventory remain separate; detection
    # never unions competing vendor roots.
    extensions = manifest.get("extensions", {}) if manifest else {}
    for dialect_name, path in (
        ("openai", ".codex-plugin/plugin.json"),
        ("cursor", ".cursor-plugin/plugin.json"),
    ):
        inline = (
            dialect_name == "openai"
            and isinstance(extensions, dict)
            and "com.openai" in extensions
        )
        overlay = "plugin.json#/extensions/com.openai" if inline else path
        if inline or path in capture.files:
            try:
                raw = extensions["com.openai"] if inline else capture.document(path)
                if not isinstance(raw, dict) or (
                    not inline and not isinstance(raw.get("name"), str)
                ):
                    raise PackageFileError("vendor_candidate_invalid")
                if not inline:
                    validate_manifest({"$schema": PLUGIN_SCHEMA, "name": raw["name"]})
                vendor_overlays[dialect_name] = raw
                candidates.append(
                    DialectCandidate(
                        dialect=dialect_name,
                        format_version="documented-2026-09-16",
                        adapter_version=(
                            "chatbook-codex/1"
                            if dialect_name == "openai"
                            else "chatbook-cursor/1"
                        ),
                        root_manifest=(
                            "plugin.json"
                            if manifest and dialect_name == "openai"
                            else path
                        ),
                        overlays=(
                            (overlay,) if manifest and dialect_name == "openai" else ()
                        ),
                        qualified=True,
                    )
                )
            except PackageFileError as exc:
                diagnostics.append(Diagnostic(code=str(exc), path=overlay))
    base = {
        "source_identity": str(capture.root),
        "candidates": tuple(candidates),
        "content_digest": capture.digest,
        "source_digest": capture.source_digest,
        "link_targets": capture.link_targets,
        "diagnostics": tuple(diagnostics),
    }
    if manifest is None and (
        "plugin.json" in capture.files
        or "plugin.json" in capture.directories
        or any(d.path == "plugin.json" for d in capture.errors)
    ):
        return PackageInspection(
            **base, rejected=True, activation_blockers=("root_manifest_invalid",)
        )
    if not candidates:
        return PackageInspection(
            **base, rejected=True, activation_blockers=("package_format_unsupported",)
        )
    if dialect is None:
        # A native portable root is the default; independent vendor roots still
        # require a choice. An OpenAI portable overlay is an alternate view.
        standalone = [
            candidate
            for candidate in candidates
            if candidate.root_manifest != "plugin.json"
        ]
        if manifest is not None and not standalone:
            dialect = "portable"
        elif len(candidates) == 1:
            dialect = candidates[0].dialect
        else:
            return PackageInspection(
                **base, activation_blockers=("dialect_choice_required",)
            )
    selected = next(
        (candidate for candidate in candidates if candidate.dialect == dialect), None
    )
    if selected is None:
        return PackageInspection(
            **base, rejected=True, activation_blockers=("dialect_unavailable",)
        )
    base.update(
        dialect=dialect,
        format_version=selected.format_version,
        adapter_version=selected.adapter_version,
        root_manifest=selected.root_manifest,
        overlay_identities=selected.overlays + ((carrier,) if carrier else ()),
    )
    vendor_records, vendor_variables, vendor_diagnostics, vendor_blockers = (
        (),
        {},
        (),
        (),
    )
    if dialect != "portable":
        from .adapters.codex import vendor_inventory

        overlay = {**(catalog_overlay or {}), **vendor_overlays[dialect]}
        try:
            vendor_records, vendor_variables, vendor_diagnostics, vendor_blockers = (
                vendor_inventory(
                    capture,
                    overlay,
                    dialect=dialect,
                    portable_root=manifest is not None and dialect == "openai",
                )
            )
        except (PackageFileError, ComponentLimitError):
            return PackageInspection(
                **base, rejected=True, activation_blockers=("vendor_inventory_invalid",)
            )

    elif catalog_overlay is not None:
        return PackageInspection(
            **base, rejected=True, activation_blockers=("catalog_dialect_unsupported",)
        )
    native_manifest = manifest or {"name": vendor_overlays[dialect]["name"]}
    try:
        inventory, edges, variables, extra, blockers, bodies = inventory_package(
            capture,
            native_manifest,
            max_components=MAX_COMPONENTS,
            vendor_records=vendor_records,
            vendor_variables=vendor_variables,
            vendor_blockers=vendor_blockers,
            discover_portable=dialect == "portable"
            or (manifest is not None and dialect == "openai"),
        )
    except (PackageFileError, ComponentLimitError) as exc:
        return PackageInspection(**base, rejected=True, activation_blockers=(str(exc),))
    diagnostics.extend(vendor_diagnostics)
    diagnostics.extend(extra)
    base["diagnostics"] = tuple(diagnostics)
    if len(inventory) > MAX_COMPONENTS:
        return PackageInspection(
            **base, rejected=True, activation_blockers=("component_count_limit",)
        )
    subject = {
        "identity": {
            key: value
            for key, value in native_manifest.items()
            if key in _MANIFEST_FIELDS - {"extensions"}
        },
        "dialect": dialect,
        "adapter": selected.adapter_version,
        "overlays": base["overlay_identities"],
        "components": {
            key: record.model_dump(exclude={"evidence"})
            for key, record in inventory.items()
        },
        "dependencies": edges,
        "variables": variables,
        "bodies": bodies,
        "blockers": blockers,
    }
    if dialect != "portable":
        subject["vendor_content_digest"] = capture.digest
    effective_digest = hashlib.sha256(canonical_json(subject).encode()).hexdigest()
    evidence = InspectionEvidence(
        revision=capture.digest,
        platform=sys.platform,
        adapter=selected.adapter_version,
        observed_at=datetime.now(UTC).isoformat(),
    )
    inventory = {
        key: record.model_copy(update={"evidence": (evidence,)})
        for key, record in inventory.items()
    }
    return PackageInspection(
        **base,
        effective_digest=effective_digest,
        inventory=inventory,
        dependency_edges=edges,
        variables_json=canonical_json(variables),
        activation_blockers=blockers,
    )
