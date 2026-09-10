"""Local backup preview and capture through installed owners and native admission."""

import hashlib
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from . import bootstrap
from .capture import (
    CaptureResult,
    CaptureReviewRequired,
    _capture_options,
    _capture_under_maintenance,
)
from .control_records import UNBOUND_NAMESPACE, admission_authority
from .inventory import discover
from .owner_registry import install_adapters
from .profile_paths import effective_config_path, lexical_path
from .storage_admission import _preview_reads


def _selectors(config_paths):
    if type(config_paths) is not tuple:
        raise TypeError("invalid_capture_selection")
    return tuple(
        lexical_path(path) for path in config_paths or (effective_config_path(),)
    )


def _review_digest(inventory, settings, selections, limits, budget):
    """Bind reviewed scope to the exact policy and budgets, not record contents."""
    document = {
        "scope": inventory.scope_digest,
        "limits": asdict(limits),
        "budget": budget,
        "credential_mode": settings["credential_mode"],
        "encrypted": settings["encrypted"],
        "allow_partial": settings["allow_partial"],
        "acknowledged_credential_issues": settings.get(
            "acknowledged_credential_issues", ()
        ),
        "selections": asdict(selections),
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, default=str).encode()
    ).hexdigest()


def preview_capture(config_paths: tuple[Path, ...], *, options):
    """Discover installed local sources without starting or pausing services."""
    settings, selections, limits, budget = _capture_options(options)
    selectors = _selectors(config_paths)
    install_adapters()
    with _preview_reads(limits=limits, byte_budget=budget):
        inventory = discover(selectors, selections=selections)
    return replace(
        inventory,
        scope_digest=_review_digest(inventory, settings, selections, limits, budget),
    )


def _capture_names(authority, inventory):
    """Reuse existing physical scopes; register only disjoint new local roots."""
    registry = bootstrap._registry(authority.control_root.parent)
    roots = {
        item.path.resolve(strict=True)
        for item in inventory.items
        if item.path is not None and item.status in {"included", "included_directory"}
    }
    names = {UNBOUND_NAMESPACE}
    for root in sorted(roots, key=lambda path: (len(path.parts), str(path))):
        covered = False
        narrower = False
        for name, entry in registry.items():
            for raw in entry["roots"]:
                registered = Path(raw).resolve(strict=True)
                if registered == root or registered in root.parents:
                    names.add(name)
                    covered = True
                elif root in registered.parents:
                    # Introducing an overlapping broader scope would invalidate
                    # live clients. Reuse descendants and consider files below.
                    names.add(name)
                    narrower = True
        if covered or narrower:
            continue
        name = "backup.source." + hashlib.sha256(str(root).encode()).hexdigest()
        authority.register(name, (root,))
        registry[name] = {"roots": [str(root)]}
        names.add(name)
    return tuple(sorted(names))


def capture(config_paths, approved_scope, destination, *, options, cancel):
    """Capture on the caller's worker thread, releasing admission before packaging.

    Live clients respond through their retained app monitor. Clients that cannot
    settle keep their native leases and cause a bounded refusal, never a forced
    close or a partial claim of complete capture.
    """
    from .archive_reader import _check
    from .archive_writer import _output
    from .space import require_capacity

    settings, selections, limits, budget = _capture_options(options)
    selectors = _selectors(config_paths)
    destination = lexical_path(destination)
    suffix = ".tldw-backup.zip.age" if settings["encrypted"] else ".tldw-backup.zip"
    if not destination.name.endswith(suffix):
        raise ValueError("invalid_backup_suffix")
    _check(cancel)
    install_adapters()
    with _preview_reads(limits=limits, byte_budget=budget):
        inventory = discover(selectors, selections=selections)
    if (
        _review_digest(inventory, settings, selections, limits, budget)
        != approved_scope
    ):
        raise CaptureReviewRequired(("scope_changed",))
    if not inventory.complete and not settings["allow_partial"]:
        raise CaptureReviewRequired(inventory.issues or ("incomplete_inventory",))
    stage_parent = Path(
        settings.get("staging_parent", Path(tempfile.gettempdir()).resolve())
    )
    _output(
        CaptureResult(stage_parent / "capture-preflight", inventory, b""), destination
    )
    estimate = sum(
        item.path.stat().st_size
        for item in inventory.items
        if item.path is not None and item.status == "included"
    )
    if estimate > budget:
        raise CaptureReviewRequired(("capture_budget_changed",))
    require_capacity(
        {
            stage_parent: estimate * 2,
            destination: estimate * (5 if settings["encrypted"] else 3),
        }
    )
    authority = admission_authority(bootstrap.default_bootstrap_root())
    names = _capture_names(authority, inventory)
    _check(cancel)
    with authority.maintenance(names, 60, cancel=cancel) as session:
        session._discover_capture_inventory(
            selectors, selections, inventory.scope_digest
        )
        return _capture_under_maintenance(
            session,
            selectors,
            inventory.scope_digest,
            destination,
            options=settings,
            cancel=cancel,
        )
