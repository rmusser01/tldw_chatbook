"""Coherent local capture contracts for the approved recovery workflow."""

from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from .inventory import discover
from .models import Inventory


@dataclass(frozen=True)
class CaptureResult:
    """Private payloads plus final original-source inventory and manifest.

    The inventory retains actual recovery control roots as intentionally excluded
    authority items, so archive publication can reject source/control aliases.
    """

    root: Path
    inventory: Inventory
    manifest_bytes: bytes


def compare_scope(approved: Inventory, current: Inventory) -> tuple[str, ...]:
    """Compare source authority independently of in-scope record/asset growth."""
    return ("scope_changed",) if approved.scope_digest != current.scope_digest else ()


class CaptureReviewRequired(ValueError):
    """A renewed review is required; issues contain only nonsecret reason codes."""

    def __init__(self, issues):
        super().__init__("capture_review_required")
        self.issues = tuple(issues)


def _manifest_for(inventory, staged, aliases, options, issues):
    import hashlib
    import json
    from datetime import datetime

    from . import archive_reader as reader
    from .owner_registry import registered

    declarations = {adapter.owner_id: adapter for adapter in registered()}
    directories, files, owners = {}, [], {}
    selected = {item.logical_id: item for item in inventory.items}
    producers = {
        item.logical_id: {
            "logical_id": item.logical_id,
            "owner_id": item.owner,
            "status": item.status,
            "dependencies": list(item.dependencies),
            "shared_group": item.shared_group,
        }
        for item in inventory.items
    }
    # An excluded owner has no observed payload schema. Its identity is still
    # needed to distinguish intentional omissions from target-only live data.
    for item in inventory.items:
        owners[item.owner] = {
            "owner_id": item.owner,
            "schema_version": 0,
            "capabilities": [],
        }
    for item in inventory.items:
        if item.status == "included_directory" and item.metadata:
            meta = item.metadata
            directories[item.logical_id] = {
                "logical_id": item.logical_id,
                "root_id": meta.root_id,
                "parent_id": meta.parent_id,
                "relative_path": meta.relative_path,
                "metadata": {
                    "version": 1,
                    "mode": meta.mode,
                    "mtime_ns": meta.mtime_ns,
                },
            }
    for item, path in staged:
        meta = item.metadata
        if meta is not None and meta.root_id in directories:
            root_id, parent_id, relative = (
                meta.root_id,
                meta.parent_id,
                meta.relative_path,
            )
        else:
            root_id = "root:" + hashlib.sha256(item.logical_id.encode()).hexdigest()
            parent_id, relative = root_id, item.path.name
            directories[root_id] = {
                "logical_id": root_id,
                "root_id": root_id,
                "parent_id": None,
                "relative_path": "",
                "metadata": {"version": 1, "mode": 0o700, "mtime_ns": 0},
                "synthetic": True,
            }
            producers[root_id] = {
                "logical_id": root_id,
                "owner_id": item.owner,
                "status": "included_directory",
                "dependencies": [],
                "shared_group": None,
            }
        producers[item.logical_id] = {
            "logical_id": item.logical_id,
            "owner_id": item.owner,
            "status": "included",
            "dependencies": list(item.dependencies),
            "shared_group": item.shared_group,
        }
        source_info = item.path.stat()
        adapter = declarations.get(item.owner)
        policy = adapter.schema_policy() if adapter else None
        owners[item.owner] = {
            "owner_id": item.owner,
            "schema_version": options["versions"].get(
                item.owner, max(policy.versions) if policy else 1
            ),
            "capabilities": [],
        }
        files.append(
            {
                "logical_id": item.logical_id,
                "root_id": root_id,
                "parent_id": parent_id,
                "relative_path": relative,
                "owner_id": item.owner,
                "payload": str(path.relative_to(options["root"])),
                "size": path.stat().st_size,
                "sha256": reader._hash(path, options["cancel"]),
                "metadata": {
                    "version": 1,
                    "mode": meta.mode if meta else source_info.st_mode & 0o777,
                    "mtime_ns": meta.mtime_ns if meta else source_info.st_mtime_ns,
                },
            }
        )
    groups = []
    represented = {item["logical_id"] for item in files} | set(directories)
    for logical_id in sorted(represented):
        item = selected.get(logical_id)
        members = {logical_id}
        if item:
            members.update(key for key in item.dependencies if key in represented)
            if item.shared_group:
                members.update(aliases[item.shared_group])
        groups.append(
            {
                "group_id": "group:" + hashlib.sha256(logical_id.encode()).hexdigest(),
                "members": sorted(members),
                "complete": True,
            }
        )
    exclusions = [
        {"logical_id": item.logical_id, "reason": item.status}
        for item in inventory.items
        if item.status not in {"included", "included_directory"}
    ]
    external = any(item.owner.startswith("external.") for item, _ in staged)
    lines = [
        "Captured files: " + str(len(files)),
        "Captured bytes: " + str(sum(item["size"] for item in files)),
    ]
    if external:
        lines.append(
            "External files have per-file stability checks, not folder-wide consistency."
        )
    if any(item.owner.startswith("diagnostics.") for item, _ in staged) or external:
        lines.append(
            "Arbitrary diagnostic and external content is not credential-sanitized."
        )
    lines.extend(issues)
    profiles = sorted(
        {
            item.logical_id.split(":")[1]
            for item in inventory.items
            if item.logical_id.startswith("profile:")
        }
    )
    doc = {
        "format_version": 1,
        "producer_version": "0.1",
        "captured_at": datetime.now(UTC).isoformat(),
        "profile_ids": profiles,
        "owners": list(owners.values()),
        "directories": list(directories.values()),
        "files": files,
        "dependency_groups": groups,
        "consistency": "partial"
        if external or issues or not inventory.complete
        else "coherent",
        "exclusions": exclusions,
        "credential_policy": options["mode"],
        "required_capabilities": [],
        "report": {"version": 1, "lines": lines},
        "relocations": [],
        "producer_inventory": list(producers.values()),
    }
    encoded = json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    if len(encoded) > options["limits"].manifest_bytes:
        raise ValueError("manifest_limit")
    reader._manifest(encoded, options["limits"], options["encrypted"])
    return encoded


def _capture_options(options):
    """Validate without IO; return defensive settings, selections, limits, budget."""
    from collections.abc import Mapping

    from .limits import ArchiveLimits
    from .models import DiscoverySelections

    if not isinstance(options, Mapping):
        raise TypeError("invalid_capture_options")
    allowed = {
        "external_roots",
        "model_ids",
        "temporary_media",
        "diagnostics",
        "allow_partial",
        "limits",
        "credential_mode",
        "encrypted",
        "byte_budget",
        "staging_parent",
        "planned_output_root",
        "acknowledged_credential_issues",
    }
    if set(options) - allowed:
        raise ValueError("unknown_capture_option")
    options = dict(options)
    for key, default in (
        ("credential_mode", "exclude"),
        ("encrypted", False),
        ("allow_partial", False),
    ):
        options.setdefault(key, default)
    for key in ("staging_parent", "planned_output_root"):
        if (
            key in options
            and options[key] is not None
            and (not isinstance(options[key], Path) or not options[key].is_absolute())
        ):
            raise TypeError("invalid_capture_options")
    if "staging_parent" in options and options["staging_parent"] is None:
        raise TypeError("invalid_capture_options")
    acknowledged = options.get("acknowledged_credential_issues", ())
    if type(acknowledged) is not tuple or any(
        type(issue) is not str for issue in acknowledged
    ):
        raise TypeError("invalid_capture_options")
    selections = DiscoverySelections(
        external_roots=options.get("external_roots", ()),
        model_ids=options.get("model_ids", ()),
        temporary_media=options.get("temporary_media", False),
        diagnostics=options.get("diagnostics", False),
        planned_output_root=options.get("planned_output_root"),
    )
    if (
        type(selections.external_roots) is not tuple
        or any(
            not isinstance(path, Path) or not path.is_absolute()
            for path in selections.external_roots
        )
        or type(selections.model_ids) is not tuple
        or any(type(model) is not str or not model for model in selections.model_ids)
        or type(selections.temporary_media) is not bool
        or type(selections.diagnostics) is not bool
        or type(options.get("allow_partial", False)) is not bool
    ):
        raise TypeError("invalid_capture_options")
    mode, encrypted = (
        options.get("credential_mode", "exclude"),
        options.get("encrypted", False),
    )
    if mode not in {"exclude", "include", "rollback"} or type(encrypted) is not bool:
        raise ValueError("invalid_capture_options")
    if mode != "exclude" and not encrypted:
        raise ValueError("credentials_require_encryption")
    limits = options.get("limits", ArchiveLimits())
    if type(limits) is not ArchiveLimits:
        raise TypeError("invalid_capture_limits")
    budget = options.get("byte_budget", limits.expanded_bytes)
    if type(budget) is not int or not 0 < budget <= limits.expanded_bytes:
        raise ValueError("invalid_capture_budget")
    return options, selections, limits, budget


def _item_validator(adapter, item):
    """Match the installed mixed recovered catalog/payload capture dispatch."""
    from .recovered_media import MAX_PAYLOAD_BYTES, _RecoveredAdapter
    from .recovery_files import _RawDeclaration

    if type(adapter) is _RecoveredAdapter and item.path.name != "catalog.sqlite3":
        return _RawDeclaration(adapter.owner_id, max_bytes=MAX_PAYLOAD_BYTES)
    return adapter


def _capture_under_maintenance(
    session, config_paths, approved_scope, destination, *, options, cancel
):
    """Stage under the caller's native-owning thread; never obtain runtime admission.

    Root orchestration must settle actual producers/participant coverage, enroll
    the complete approved namespace closure and acquire native maintenance before
    entry. Registry adapters must be installed before preview and remain stable.
    Return occurs before caller releases maintenance, with no packaging/encryption.
    """
    import hashlib
    import os
    import shutil
    import tempfile
    from dataclasses import replace
    from uuid import uuid4

    from . import archive_reader as reader
    from .archive_writer import _output
    from .credentials import process_credentials
    from .models import StorageItem
    from .native_files import create_private_directory, create_private_file
    from .owner_registry import registered
    from .space import require_capacity
    from .sqlite_validation import validated_schema_version
    from .storage_admission import MaintenanceSession

    if type(session) is not MaintenanceSession:
        raise ValueError("native_maintenance_required")
    session._check()
    reader._check(cancel)
    options, selections, limits, budget = _capture_options(options)
    mode, encrypted = options["credential_mode"], options["encrypted"]
    adapters = {adapter.owner_id: adapter for adapter in registered()}
    with session._discovery_reads():
        current = discover(config_paths, selections=selections)
    if current.scope_digest != approved_scope:
        raise CaptureReviewRequired(("scope_changed",))
    if not current.complete and (
        not options.get("allow_partial", False)
        or set(current.issues)
        - {"unsupported", "unavailable", "missing_required", "unsupported_owner"}
    ):
        raise CaptureReviewRequired(current.issues or ("incomplete_inventory",))
    entries = tuple(item for item in current.items if item.status == "included")
    if not entries or any(
        item.path is None or item.owner not in adapters for item in entries
    ):
        raise ValueError("capture_owner_unavailable")
    estimate = sum(item.path.stat().st_size for item in entries)
    if estimate > budget or len(current.items) + 1 > limits.members:
        raise CaptureReviewRequired(("capture_budget_changed",))
    parent = Path(options.get("staging_parent", Path(tempfile.gettempdir()).resolve()))
    stage = parent / ("capture-" + uuid4().hex)
    # Preserve the actual used control root solely as output-alias authority.
    control = StorageItem(
        "recovery.control",
        "control:" + hashlib.sha256(str(session._control).encode()).hexdigest(),
        session._control.parent,
        "intentionally_excluded",
        (),
    )
    final_inventory = replace(current, items=(*current.items, control))
    _output(CaptureResult(stage, final_inventory, b""), Path(destination))
    require_capacity(
        {parent: estimate * 2, Path(destination): estimate * (5 if encrypted else 3)}
    )
    create_private_directory(stage)
    completed = False
    try:
        create_private_directory(stage / "payload")
        sources = tuple(dict.fromkeys(item.path for item in entries))
        staged, physical, aliases, versions = [], {}, {}, {}
        with session.capture_scope(sources, stage, limits=limits, byte_budget=budget):
            for item in entries:
                reader._check(cancel)
                path = (
                    stage
                    / "payload"
                    / hashlib.sha256(item.logical_id.encode()).hexdigest()
                )
                info = item.path.stat()
                key = info.st_dev, info.st_ino
                adapter = adapters[item.owner]
                if key in physical:
                    previous, previous_item = physical[key]
                    if (
                        not item.shared_group
                        or item.shared_group != previous_item.shared_group
                    ):
                        raise ValueError("undeclared_alias")
                    with (
                        reader._regular(previous) as source,
                        create_private_file(path) as descriptor,
                    ):
                        while chunk := source.read(1024**2):
                            reader._check(cancel)
                            view = memoryview(chunk)
                            while view:
                                written = os.write(descriptor, view)
                                if not written:
                                    raise OSError("capture_write_failed")
                                view = view[written:]
                else:
                    for attempt in range(3):
                        try:
                            adapter.capture(item, path, cancel)
                            break
                        except (OSError, ValueError):
                            if not item.owner.startswith("external.") or attempt == 2:
                                raise
                            path.unlink(missing_ok=True)
                            reader._check(cancel)
                    physical[key] = path, item
                staged.append((item, path))
                if item.shared_group:
                    aliases.setdefault(item.shared_group, []).append(item.logical_id)
                validator = _item_validator(adapter, item)
                policy = validator.schema_policy()
                if policy and policy.schema_sql:
                    observed = validated_schema_version(validator, path, cancel)
                    if item.owner in versions and versions[item.owner] != observed:
                        raise CaptureReviewRequired(("mixed_owner_schema_versions",))
                    versions[item.owner] = observed
                else:
                    validation = validator.validate(path)
                    if validation:
                        raise CaptureReviewRequired(validation)
                total = sum(candidate.stat().st_size for _, candidate in staged)
                if total > budget or path.stat().st_size > limits.member_bytes:
                    raise CaptureReviewRequired(("capture_budget_changed",))
                require_capacity(
                    {stage: total, Path(destination): total * (5 if encrypted else 3)}
                )
            from .config_adapter import _ChatbookRegistry

            for item, path in staged:
                adapter = adapters[item.owner]
                if type(adapter) is _ChatbookRegistry:
                    adapter.prepare_capture(item, path, entries)
            candidates = {item.logical_id: path for item, path in staged}
            from .rag_projection_validation import validate_groups

            validate_groups(current.items, candidates, stage, cancel, limits, budget)
            for item, path in staged:
                adapter = adapters[item.owner]
                if (
                    item.owner == "recovered.media"
                    and _item_validator(adapter, item) is adapter
                ):
                    dependency_issues = adapter.validate_dependencies(
                        item, path, candidates
                    )
                    if dependency_issues:
                        raise CaptureReviewRequired(dependency_issues)
            rebound = replace(
                current, items=tuple(replace(item, path=path) for item, path in staged)
            )
            issues = process_credentials(stage, rebound, mode=mode, encrypted=encrypted)
            acknowledged = options.get("acknowledged_credential_issues", ())
            if (
                type(acknowledged) is not tuple
                or mode == "exclude"
                and issues
                or set(issues) != set(acknowledged)
            ):
                raise CaptureReviewRequired(issues or ("credential_coverage_changed",))
            if mode != "exclude":
                material = stage / "credential-recovery.json"
                target = stage / "payload" / "credential-recovery.json"
                from .credentials import _read, _write

                _write(target, _read(material).decode("utf-8"))
                # Material references payload names; its own archive location is
                # fixed and installed, not a live destination or credential scope.
                material_item = StorageItem(
                    "recovery.credentials", "credentials", material, "included", ()
                )
                staged.append((material_item, target))
            with session._discovery_reads():
                observed_inventory = discover(config_paths, selections=selections)
            if (
                observed_inventory.scope_digest != approved_scope
                or tuple(adapters.values()) != registered()
            ):
                raise CaptureReviewRequired(("scope_changed",))
            if {
                item.logical_id
                for item in observed_inventory.items
                if item.status == "included"
            } != {item.logical_id for item in entries}:
                raise CaptureReviewRequired(("capture_boundary_changed",))
            if issues:
                final_inventory = replace(
                    final_inventory,
                    complete=False,
                    issues=tuple(sorted(set(final_inventory.issues) | set(issues))),
                )
            total = sum(path.stat().st_size for _, path in staged)
            if total > budget:
                raise CaptureReviewRequired(("capture_budget_changed",))
            manifest = _manifest_for(
                final_inventory,
                staged,
                aliases,
                {
                    "root": stage,
                    "cancel": cancel,
                    "mode": mode,
                    "encrypted": encrypted,
                    "versions": versions,
                    "limits": limits,
                },
                tuple(sorted(set(issues) | set(current.issues))),
            )
            reader._check(cancel)
        completed = True
        return CaptureResult(stage, final_inventory, manifest)
    finally:
        if not completed:
            shutil.rmtree(stage)


def capture(config_paths, approved_scope, destination, *, options, cancel):
    """Capture through the installed runtime/native orchestration boundary."""
    from .capture_service import capture as capture_service

    return capture_service(
        config_paths, approved_scope, destination, options=options, cancel=cancel
    )
