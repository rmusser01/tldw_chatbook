"""Prove final Settings still locate the reviewed unselected local owner data."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .data_groups import group_for_owner
from .models import DISCOVERY_CONTEXT_KEY, DiscoveryContext, DiscoverySelections
from .owner_registry import install_adapters


def _read_scope(items, target):
    """Require existing bounded preview or native-held discovery authority."""
    from . import bootstrap
    from .control_records import UNBOUND_NAMESPACE
    from .storage_admission import (
        _config_capture_item,
        _contains_capture_path,
        _DiscoveryScope,
        _local,
        _PreviewScope,
    )

    scope = getattr(_local, "discovery_scope", None) or getattr(
        _local, "preview_scope", None
    )
    if type(scope) not in {_DiscoveryScope, _PreviewScope}:
        raise ValueError("preserved_group_read_scope_required")
    scope.check()
    if type(scope) is _DiscoveryScope and (
        scope.session._control != bootstrap.default_bootstrap_root() / "admission"
        or UNBOUND_NAMESPACE not in scope.session._names
        or any(
            not _contains_capture_path(scope.session._roots, item.path)
            and not _config_capture_item(item, target, scope.config_sources)
            for item in items
        )
    ):
        raise ValueError("preserved_group_native_scope_required")
    return scope


def _absence_names(registry, paths):
    """Find existing namespaces for metadata-only probes of absent SQL paths."""
    from .bootstrap import _overlap

    return {
        name
        for name, entry in registry.items()
        if any(
            _overlap(path, Path(raw))
            for raw in (
                *entry["roots"],
                *entry.get("proposed", ()),
                *(
                    token[5:]
                    for token in entry.get("historical", ())
                    if token.startswith("path:")
                ),
            )
            for path in paths
        )
    }


def _check_absences(items, owners, scope):
    """Recheck absent stores while leaving all actual content reads confined."""
    from . import bootstrap
    from .restore_plan import _absent_sqlite_paths
    from .storage_admission import _DiscoveryScope

    scope.check()
    paths = []
    for item in items:
        absent = _absent_sqlite_paths(item, owners.get(item.owner))
        if absent is None:
            raise ValueError(
                "preserved_group_path_changed:" + group_for_owner(item.owner)
            )
        paths.extend(absent)
    if type(scope) is _DiscoveryScope:
        registry = bootstrap._registry(bootstrap.default_bootstrap_root())
        if not _absence_names(registry, paths) <= set(scope.session._names):
            raise ValueError("preserved_group_native_scope_required")
        return registry
    return None


def _path_evidence(item):
    """Compare raw owner reachability, independently of inventory graph labels.

    Full inventory merges shared cohorts and expands cross-profile dependencies.
    Those labels are not raw adapter evidence. The reviewed graph retains them;
    ordinary plan/object rechecks bind preserved bytes and physical identities.
    """
    metadata = item.metadata
    return (
        item.logical_id,
        item.path,
        item.status,
        None
        if metadata is None
        else (
            metadata.root_id,
            metadata.parent_id,
            metadata.relative_path,
            metadata.kind,
            metadata.policy,
        ),
    )


def preserved_settings_fingerprint(plan) -> str:
    """Recheck staged preservation without reading a pending owner generation."""
    from .restore_plan import _fingerprint

    if plan.target is None or set(plan.target.issues) - {
        "missing_required", "dependency_unavailable"
    }:
        raise ValueError("preserved_group_inventory_invalid")
    by_id = {item.logical_id: item for item in plan.target.items}
    items = []
    for logical_id, path in plan.preserve:
        item = by_id.get(logical_id)
        if item is None or item.path != path or path is None:
            raise ValueError("preserved_group_inventory_invalid")
        items.append(item)
    owners = {owner.owner_id: owner for owner in install_adapters()}
    absent = tuple(
        item for item in items
        if item.status in {"unused", "missing_required"}
        and (owner := owners.get(item.owner)) is not None
        and (policy := owner.schema_policy()) is not None
        and policy.schema_sql
    )
    scope = _read_scope(
        tuple(item for item in items if item.status in {"included", "included_directory"}),
        plan.target,
    )
    registry = _check_absences(absent, owners, scope)
    paths = {path for _, path in plan.preserve}
    for item in items:
        owner = owners.get(item.owner)
        policy = owner.schema_policy() if owner is not None else None
        role = getattr(owner, "restore_role", None)
        if (
            item.status == "included" and policy is not None and policy.schema_sql
            and (not callable(role) or role(item) == "sqlite")
        ):
            # An absent sidecar may not have an inventory row. Its arrival still
            # changes this preserved installed SQL store's reviewed footprint.
            paths.update(Path(str(item.path) + suffix) for suffix in ("-wal", "-shm", "-journal"))
    fingerprint = _fingerprint(tuple(paths), plan.target)
    if _check_absences(absent, owners, scope) != registry:
        raise ValueError("preserved_group_native_scope_required")
    return fingerprint


def check_preserved_group_paths(plan, final_config: Mapping, config_path: Path) -> None:
    """Rediscover unselected data using final config under the caller's read scope.

    Call after config relocation/profile naming, within ``_preview_reads()`` or
    ``session._discovery_reads()``. Staging binds this proof to the candidate and
    preserved-path fingerprint; finalization rechecks those without ordinary
    owner discovery during pending activation. This grants no storage authority.
    """
    if plan.mode != "replace" or "settings" not in plan.effective_groups:
        return
    if (
        plan.target is None
        or not isinstance(config_path, Path)
        or not config_path.is_absolute()
        or ".." in config_path.parts
        or not isinstance(final_config, Mapping)
        or DISCOVERY_CONTEXT_KEY in final_config
    ):
        raise ValueError("preserved_group_config_invalid")
    if set(plan.target.issues) - {"missing_required", "dependency_unavailable"}:
        raise ValueError("preserved_group_inventory_invalid")
    profile = hashlib.sha256(str(config_path).encode()).hexdigest()[:24]
    config_id = f"profile:{profile}:config"
    configs = [
        item
        for item in plan.target.items
        if item.owner == "config"
        and item.path == config_path
        and item.logical_id == config_id
        and item.status == "included"
    ]
    if len(configs) != 1 or not any(
        path == config_path
        and len(key.split(":")) == 3
        and key.startswith("profile:")
        and key.endswith(":config")
        for key, path in plan.restore
    ):
        raise ValueError("preserved_group_config_invalid")
    owners = {owner.owner_id: owner for owner in install_adapters()}
    preserved = [
        item
        for item in plan.target.items
        if item.logical_id.startswith(f"profile:{profile}:")
        and (group := group_for_owner(item.owner)) is not None
        and group not in plan.effective_groups
    ]
    items = []
    for item in preserved:
        owner = owners.get(item.owner)
        policy = owner.schema_policy() if owner is not None else None
        if item.status in {"included", "included_directory"} or (
            item.status in {"unused", "missing_required"}
            and policy is not None
            and policy.schema_sql
        ):
            items.append(item)
        elif item.status in {"unsupported", "unavailable", "missing_required"}:
            raise ValueError(
                "preserved_group_path_changed:" + group_for_owner(item.owner)
            )
    if not items:
        return
    if any(item.path is None for item in items):
        raise ValueError("preserved_group_inventory_invalid")
    absent = tuple(
        item for item in items if item.status in {"unused", "missing_required"}
    )
    scope = _read_scope(
        (*configs, *(item for item in items if item not in absent)), plan.target
    )
    absence_registry = _check_absences(absent, owners, scope)
    grouped = {}
    for item in items:
        # These roots come from explicit local folder selections, independent
        # of configuration. Settings cannot change their selected locator.
        if item.owner != "external.files":
            grouped.setdefault(item.owner, []).append(item)
    for owner_id, expected in grouped.items():
        group = group_for_owner(owner_id)
        owner = owners.get(owner_id)
        if owner is None:
            raise ValueError("preserved_group_owner_unverified:" + group)
        try:
            model_ids = ()
            if owner_id == "models.artifacts":
                # Baseline inventory includes installed model descriptors even
                # when model payloads were omitted. Recover only their real IDs;
                # rediscovery here does not add any payload to a restore plan.
                from tldw_chatbook.Model_Artifacts.recovery import _Artifacts

                if type(owner) is not _Artifacts:
                    raise ValueError("model_owner_unverified")
                model_ids = tuple(
                    sorted(
                        {
                            owner._descriptor(item.path).model_id
                            for item in expected
                            if item.status == "included"
                            and item.path.name == "manifest.json"
                            and item.metadata is not None
                            and set(item.dependencies)
                            - {config_id, item.metadata.parent_id}
                        }
                    )
                )
            configured = dict(final_config)
            configured[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
                config_path,
                profile,
                DiscoverySelections(model_ids=model_ids, temporary_media=True),
            )
            discovered = owner.discover(configured)
            actual = {}
            for item in discovered:
                if item.owner != owner_id or item.logical_id in actual:
                    raise ValueError("invalid_owner_evidence")
                actual[item.logical_id] = item
            if any(
                item.logical_id not in actual
                or _path_evidence(actual[item.logical_id]) != _path_evidence(item)
                for item in expected
            ):
                raise ValueError("owner_path_changed")
        except (OSError, ValueError, TypeError, RuntimeError, RecursionError):
            raise ValueError("preserved_group_path_changed:" + group) from None
    if _check_absences(absent, owners, scope) != absence_registry:
        raise ValueError("preserved_group_native_scope_required")
