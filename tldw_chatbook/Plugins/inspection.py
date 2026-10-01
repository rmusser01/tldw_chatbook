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
)
from .schemas import NAMESPACE, validate_manifest

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


def inspect_package(
    root: Path, *, dialect: str | None = None, catalog_overlay: dict | None = None
) -> PackageInspection:
    """Inspect a bounded package without fetching, executing, or granting trust.

    Args:
        root: Existing canonical absolute package directory.
        dialect: Explicit portable, openai, or cursor interpretation.
        catalog_overlay: Executable vendor catalog fields; identical retained bytes
            are required before activation.

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
    capture: PackageCapture, *, dialect: str | None = None, catalog_overlay: dict | None = None
) -> PackageInspection:
    """Interpret exactly the captured bytes (also used by materialization)."""
    diagnostics = list(capture.diagnostics) + list(capture.errors)
    candidates = []
    manifest = None
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

    # Vendor views use pinned interpretation adapters; parsing never grants use.
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
                candidates.append(
                    DialectCandidate(
                        dialect=dialect_name,
                        root_manifest=(
                            "plugin.json"
                            if manifest and dialect_name == "openai"
                            else path
                        ),
                        overlays=(
                            (overlay,) if manifest and dialect_name == "openai" else ()
                        ),
                        format_version="portable/1.0.0"
                        if manifest and dialect_name == "openai"
                        else "docs/2026-10-01",
                        adapter_version="chatbook-" + dialect_name + "/2026-10-01.1",
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
        overlay_identities=selected.overlays,
    )
    if dialect != "portable":
        from .adapters.codex import overlay_for_openai
        from .adapters.vendor import CATALOG_PATH, catalog_input, inventory_vendor

        try:
            catalog, retained = catalog_input(capture, catalog_overlay)
            if dialect == "openai" and manifest is not None:
                compatibility = (
                    capture.document(".codex-plugin/plugin.json")
                    if ".codex-plugin/plugin.json" in capture.files
                    and "com.openai" not in extensions
                    else None
                )
                overlay = overlay_for_openai(
                    extensions.get("com.openai"), compatibility
                )
                identity = {
                    key: value
                    for key, value in manifest.items()
                    if key in _MANIFEST_FIELDS - {"extensions"}
                }
                declaration = {**catalog, **overlay, **identity}
                portable = manifest
            else:
                declaration = {**catalog, **capture.document(selected.root_manifest)}
                portable = None
            inventory, edges, variables, extra, blockers, bodies, identity = (
                inventory_vendor(
                    capture,
                    declaration,
                    dialect,
                    max_components=MAX_COMPONENTS,
                    portable_manifest=portable,
                )
            )
            if not retained:
                blockers = tuple(sorted({*blockers, "catalog_overlay_not_retained"}))
            overlays = tuple(
                dict.fromkeys(
                    (*selected.overlays, *((CATALOG_PATH,) if catalog else ()))
                )
            )
            subject = {
                "identity": identity,
                "dialect": dialect,
                "adapter": selected.adapter_version,
                "declaration": declaration,
                "overlays": overlays,
                "content_digest": capture.digest,
                "components": {
                    key: row.model_dump(exclude={"evidence"})
                    for key, row in inventory.items()
                },
                "dependencies": edges,
                "variables": variables,
                "blockers": blockers,
                "bodies": bodies,
            }
            evidence = InspectionEvidence(
                adapter=selected.adapter_version,
                revision=capture.digest,
                platform=sys.platform,
                observed_at=datetime.now(UTC).isoformat(),
            )
            inventory = {
                key: row.model_copy(update={"evidence": (evidence,)})
                for key, row in inventory.items()
            }
            base.update(diagnostics=(*diagnostics, *extra), overlay_identities=overlays)
            return PackageInspection(
                **base,
                inventory=inventory,
                dependency_edges=edges,
                variables_json=canonical_json(variables),
                activation_blockers=blockers,
                effective_digest=hashlib.sha256(
                    canonical_json(subject).encode()
                ).hexdigest(),
            )
        except (PackageFileError, ComponentLimitError, TypeError, ValueError) as error:
            code = (
                str(error)
                if isinstance(error, (PackageFileError, ComponentLimitError))
                else "vendor_interpretation_invalid"
            )
            return PackageInspection(**base, rejected=True, activation_blockers=(code,))
    try:
        inventory, edges, variables, extra, blockers, bodies = inventory_package(
            capture, manifest, max_components=MAX_COMPONENTS
        )
    except (PackageFileError, ComponentLimitError) as exc:
        return PackageInspection(**base, rejected=True, activation_blockers=(str(exc),))
    diagnostics.extend(extra)
    base["diagnostics"] = tuple(diagnostics)
    if len(inventory) > MAX_COMPONENTS:
        return PackageInspection(
            **base, rejected=True, activation_blockers=("component_count_limit",)
        )
    subject = {
        "identity": {
            key: value
            for key, value in manifest.items()
            if key in _MANIFEST_FIELDS - {"extensions"}
        },
        "dialect": dialect,
        "adapter": selected.adapter_version,
        "overlays": selected.overlays,
        "components": {
            key: record.model_dump(exclude={"evidence"})
            for key, record in inventory.items()
        },
        "dependencies": edges,
        "variables": variables,
        "bodies": bodies,
        "blockers": blockers,
    }
    effective_digest = hashlib.sha256(canonical_json(subject).encode()).hexdigest()
    evidence = InspectionEvidence(
        revision=capture.digest,
        platform=sys.platform,
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
